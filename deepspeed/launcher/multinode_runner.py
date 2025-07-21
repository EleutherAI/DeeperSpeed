# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import json
import os
import sys
import shutil
import subprocess
import warnings
from shlex import split
from abc import ABC, abstractmethod
from deepspeed.accelerator import get_accelerator
from ..utils import logger, get_numactl_cmd
from .constants import PDSH_MAX_FAN_OUT, MVAPICH_TMP_HOSTFILE


class MultiNodeRunner(ABC):

    def __init__(self, args, world_info_base64):
        self.args = args
        self.validate_args()
        self.user_arguments = self.parse_user_args()
        self.user_script = args.user_script
        self.world_info_base64 = world_info_base64
        self.exports = {}

    @abstractmethod
    def backend_exists(self):
        """Return whether the corresponding backend exists"""

    @abstractmethod
    def get_cmd(self, environment, active_resources):
        """Return the command to execute on node"""

    def add_export(self, key, var):
        self.exports[key.strip()] = f"\"{var.strip()}\""

    def parse_user_args(self):
        return self.args.user_args

    @property
    def name(self):
        """Return the name of the backend"""
        return self.__class__.__name__

    def validate_args(self):
        """Validate self.args"""


class PDSHRunner(MultiNodeRunner):

    def __init__(self, args, world_info_base64):
        super().__init__(args, world_info_base64)

    def backend_exists(self):
        return shutil.which('pdsh')

    def parse_user_args(self):
        processed_args = []
        for arg in self.args.user_args:
            # With pdsh, if we are passing a string as an argument, it will get
            # split on whitespace. To avoid this and support strings that
            # contain '"', we do this extra processing step:
            if " " in arg:
                arg = '"{}"'.format(arg.replace('"', '\\"'))
            processed_args.append(arg)
        return processed_args

    @property
    def name(self):
        return "pdsh"

    def get_cmd(self, environment, active_resources):
        environment['PDSH_RCMD_TYPE'] = 'ssh'
        if self.args.ssh_port is not None:  # only specify ssh port if it is specified
            environment["PDSH_SSH_ARGS_APPEND"] = f"{environment.get('PDSH_SSH_ARGS_APPEND', '')} \
            -p {self.args.ssh_port}"

        active_workers = ",".join(active_resources.keys())
        logger.info("Running on the following workers: %s" % active_workers)

        # PDSH flags for max node fan out and specific hosts to launch on
        # See https://linux.die.net/man/1/pdsh for flag details
        pdsh_cmd_args = ['pdsh', '-S', '-f', str(PDSH_MAX_FAN_OUT), '-w', active_workers] + split(
            self.args.launcher_args)

        exports = ""
        for key, val in self.exports.items():
            exports += "export {}={}; ".format(key, val)

        # https://linux.die.net/man/1/pdsh
        # %n will be replaced by pdsh command
        deepspeed_launch = [
            exports, f"cd {os.path.abspath('.')};", sys.executable, "-u", "-m", "deepspeed.launcher.launch",
            f'--world_info={self.world_info_base64}', "--node_rank=%n", f"--master_addr={self.args.master_addr}",
            f"--master_port={self.args.master_port}"
        ]
        if self.args.no_python:
            deepspeed_launch.append("--no_python")
        if self.args.module:
            deepspeed_launch.append("--module")
        if self.args.no_local_rank:
            deepspeed_launch.append("--no_local_rank")
        if self.args.save_pid:
            deepspeed_launch += ["--save_pid", f"{os.getpid()}"]
        if self.args.enable_each_rank_log:
            deepspeed_launch.append(f"--enable_each_rank_log={self.args.enable_each_rank_log}")
        if self.args.elastic_training:
            deepspeed_launch.append("--enable_elastic_training")
            deepspeed_launch.append(f"--max_elastic_nodes={self.args.max_elastic_nodes}")
            deepspeed_launch.append(f"--min_elastic_nodes={self.args.min_elastic_nodes}")

        cmd_to_search = [i + "\\" for i in deepspeed_launch[2:6]]

        kill_command = pdsh_cmd_args + ["pkill -f ", " ".join(cmd_to_search)[:-2]]
        return pdsh_cmd_args + deepspeed_launch + [self.user_script] + self.user_arguments, kill_command, environment


class OpenMPIRunner(MultiNodeRunner):

    def __init__(self, args, world_info_base64, resource_pool):
        super().__init__(args, world_info_base64)
        self.resource_pool = resource_pool
        self.add_export('UCX_TLS', 'tcp')

    def backend_exists(self):
        #TODO: if IB is available we should suggestion mvapich
        return shutil.which('ompi_info')

    @property
    def name(self):
        return "openmpi"

    def validate_args(self):
        super().validate_args()

        # Validate and set MPI environment variables
        self._setup_mpi_environment()

        #TODO: Allow for include/exclude at node-level but not gpu-level
        if self.args.include != "" or self.args.exclude != "":
            raise ValueError(f"{self.name} backend does not support worker include/exclusion")
        if self.args.num_nodes != -1 or self.args.num_gpus != -1:
            raise ValueError(f"{self.name} backend does not support limiting num nodes/gpus")

    def _setup_mpi_environment(self):
        """Sets up MPI-related environment variables or raises an error if they're missing."""

        required_vars = ['OMPI_COMM_WORLD_LOCAL_RANK', 'OMPI_COMM_WORLD_RANK', 'OMPI_COMM_WORLD_SIZE']

        # Check if all these are present
        if not all(var in os.environ for var in required_vars):
            raise EnvironmentError("MPI environment variables are not set. "
                                   "Ensure you are running the script with an MPI-compatible launcher.")

        # Now safe to read all
        os.environ['LOCAL_RANK'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
        os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
        os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']

    def get_cmd(self, environment, active_resources):
        total_process_count = sum(self.resource_pool.values())

        launcher_args = split(self.args.launcher_args)

        # If btl_tcp_if_include option is provided through launcher_args, we use it. Otherwise, we add
        # `--mca btl_tcp_if_include eth0` option as a default value for compatibility.
        btl_tcp_opt = ['--mca', 'btl_tcp_if_include', 'eth0']
        if len(launcher_args) >= 2:
            for i in range(len(launcher_args) - 1):
                if launcher_args[i] in ['-mca', '--mca'] and launcher_args[i + 1] == 'btl_tcp_if_include':
                    btl_tcp_opt = []
                    break

        mpirun_cmd = [
            'mpirun',
            '-n',
            f'{total_process_count}',
            '-hostfile',
            f'{self.args.hostfile}',
            '--mca',
            'btl',
            '^openib',
        ] + btl_tcp_opt + launcher_args

        export_cmd = []
        for k, v in self.exports.items():
            export_cmd += ['-x', "{}={}".format(k, v)]

        python_exec = []
        if not self.args.no_python:
            python_exec = [sys.executable, "-u"]
            if self.args.module:
                python_exec.append("-m")

        return mpirun_cmd + export_cmd + python_exec + [self.user_script] + self.user_arguments


class JSRunner(MultiNodeRunner):
    def __init__(self, args, world_info_base64, resource_pool):
        super().__init__(args, world_info_base64)
        self.resource_pool = resource_pool
        self.add_export('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5')

    def backend_exists(self):
        #TODO: if IB is available we should suggestion mvapich
        #This ompi check will still work for jsrun since spectrum-mpi is based on ompi
        return shutil.which('ompi_info')

    @property
    def name(self):
        return "jsrun"

    def validate_args(self):
        super().validate_args()
        #TODO: Allow for include/exclude at node-level but not gpu-level
        if self.args.include != "" or self.args.exclude != "":
            raise ValueError(
                f"{self.name} backend does not support worker include/exclusion")
        if self.args.num_nodes != -1 or self.args.num_gpus != -1:
            raise ValueError(
                f"{self.name} backend does not support limiting num nodes/gpus")

    def get_cmd(self, environment, active_resources):
        total_process_count = sum(self.resource_pool.values())

        jsrun_cmd = [
            'jsrun',
            '-n',
            f'{total_process_count}',
            '-c',
            f'{7}',
            '-g',
            f'{1}',
            '-a',
            f'{1}',

        ] + split(self.args.launcher_args)

        export_cmd = []
        for k, v in self.exports.items():
            export_cmd += ['-E', "{}={}".format(k, v)]

        python_exec = []
        if not self.args.no_python:
            python_exec = [sys.executable, "-u"]
            if self.args.module:
                python_exec.append("-m")

        return jsrun_cmd + export_cmd + python_exec + [self.user_script
                                                        ] + self.user_arguments


class MPICHRunner(MultiNodeRunner):

    def __init__(self, args, world_info_base64, resource_pool):
        super().__init__(args, world_info_base64)
        self.resource_pool = resource_pool

    def backend_exists(self):
        #TODO: if IB is available we should suggestion mpich
        return shutil.which('mpirun')  #mpich_info

    @property
    def name(self):
        return "mpich"

    def validate_args(self):
        super().validate_args()
        #TODO: Allow for include/exclude at node-level but not gpu-level
        if self.args.include != "" or self.args.exclude != "":
            raise ValueError(f"{self.name} backend does not support worker include/exclusion")

        if self.args.num_nodes != -1 or self.args.num_gpus != -1:
            raise ValueError(f"{self.name} backend does not support limiting num nodes/gpus")

    def get_cmd(self, environment, active_resources):
        devices_per_node = self.resource_pool.values()
        total_process_count = sum(devices_per_node)
        process_per_node = list(devices_per_node)[0]
        if not all([n == process_per_node for n in devices_per_node]):
            raise ValueError("MPICH requires same number of devices per node")

        mpirun_cmd = [
            'mpirun',
            '-n',
            f'{total_process_count}',
            '-ppn',
            f'{process_per_node}',
        ] + split(self.args.launcher_args)
        export_cmd = []

        for k, v in self.exports.items():
            export_cmd += ['-genv', "{}={}".format(k, v)]

        export_cmd += ['-genv', 'MASTER_ADDR', str(self.args.master_addr)]
        export_cmd += ['-genv', 'MASTER_PORT', str(self.args.master_port)]
        export_cmd += ['-genv', 'WORLD_SIZE', str(total_process_count)]
        export_cmd += ['-genv', 'LOCAL_SIZE', str(process_per_node)]

        export_cmd += ['-hosts']
        hosts = ""
        for i, host in enumerate(self.resource_pool.keys()):
            if i == 0:
                hosts = f"{host}"
            else:
                hosts += f",{host}"
        export_cmd += [hosts]

        helper_args = ["--launcher"] + [self.args.launcher]
        python_exec = []
        if not self.args.no_python:
            python_exec += [sys.executable, "-u"]
            if self.args.module:
                python_exec.append("-m")
                helper_args.append("--module")
        else:
            helper_args.append("--no_python")

        helper_cmd = str(os.path.dirname(os.path.realpath(__file__))) + '/launcher_helper.py'
        helper_cmd = [helper_cmd] + helper_args + [self.user_script] + self.user_arguments

        return mpirun_cmd + export_cmd + python_exec + helper_cmd


class IMPIRunner(MultiNodeRunner):

    def __init__(self, args, world_info_base64, resource_pool):
        super().__init__(args, world_info_base64)
        self.resource_pool = resource_pool

    def backend_exists(self):
        #TODO: if IB is available we should suggestion mpich
        return shutil.which('mpirun')  #mpich_info

    @property
    def name(self):
        return "impi"

    def validate_args(self):
        super().validate_args()
        #TODO: Allow for include/exclude at node-level but not gpu-level
        if self.args.include != "" or self.args.exclude != "":
            raise ValueError(f"{self.name} backend does not support worker include/exclusion")

        if self.args.num_nodes != -1 or self.args.num_gpus != -1:
            raise ValueError(f"{self.name} backend does not support limiting num nodes/gpus")

    def get_cmd(self, environment, active_resources):
        devices_per_node = self.resource_pool.values()
        total_process_count = sum(devices_per_node)
        process_per_node = list(devices_per_node)[0]
        if not all([n == process_per_node for n in devices_per_node]):
            raise ValueError("Intel MPI requires same number of devices per node")

        mpirun_cmd = [
            'mpirun',
            '-ppn',
            f'{process_per_node}',
        ] + split(self.args.launcher_args)
        export_cmd = []

        for k, v in self.exports.items():
            export_cmd += ['-genv', f'{k}', f'{v}']

        if self.args.bind_cores_to_rank:
            cores_per_rank, _ = get_numactl_cmd(self.args.bind_core_list, process_per_node, 0)
            export_cmd += ['-genv', 'OMP_NUM_THREADS', str(cores_per_rank)]

        export_cmd += ['-genv', 'MASTER_ADDR', str(self.args.master_addr)]
        export_cmd += ['-genv', 'MASTER_PORT', str(self.args.master_port)]
        export_cmd += ['-genv', 'WORLD_SIZE', str(total_process_count)]
        export_cmd += ['-genv', 'LOCAL_SIZE', str(process_per_node)]

        # turn off IMPI core binding, use deepspeed's own core binding
        export_cmd += ['-genv', 'I_MPI_PIN', '0']

        export_cmd += ['-hosts']
        hosts = ""
        for i, host in enumerate(self.resource_pool.keys()):
            if i == 0:
                hosts = f"{host}"
            else:
                hosts += f",{host}"
        export_cmd += [hosts]

        per_host_cmd = []

        for i in range(total_process_count):
            local_rank = i % process_per_node
            python_exec = []
            if self.args.bind_cores_to_rank:
                _, numactl_cmd = get_numactl_cmd(self.args.bind_core_list, process_per_node, local_rank)
                python_exec += numactl_cmd

            if not self.args.no_python:
                python_exec += [sys.executable, "-u"]
                if self.args.module:
                    python_exec.append("-m")
            env_mapping = ['-env', 'RANK', str(i)]
            env_mapping += ['-env', 'LOCAL_RANK', str(local_rank)]
            if i == 0:
                per_host_cmd = ['-n', '1'] + env_mapping + python_exec + [self.user_script] + self.user_arguments
            else:
                per_host_cmd = per_host_cmd + [':', '-n', '1'] + env_mapping + python_exec + [self.user_script
                                                                                              ] + self.user_arguments
        print(mpirun_cmd + export_cmd + per_host_cmd)
        return mpirun_cmd + export_cmd + per_host_cmd


class SlurmRunner(MultiNodeRunner):

    def __init__(self, args, world_info_base64, resource_pool):
        super().__init__(args, world_info_base64)
        self.resource_pool = resource_pool

    def backend_exists(self):
        return shutil.which('sinfo')

    def parse_user_args(self):
        user_args = []
        for arg in self.args.user_args:
            if arg.startswith('{') and arg.endswith('}'):
                try:
                    arg_dict = json.loads(arg)
                    if 'config_files' in arg_dict:
                        config_files = {}
                        for k, v in arg_dict.get('config_files', {}).items():
                            config_files[k] = json.loads(v)
                        arg_dict['config_files'] = config_files
                except json.JSONDecodeError as jde:
                    raise ValueError(
                        'SLURM is picky and needs you to use plain json for your configs. Check for comments and lowercase trues'
                    ) from jde
                arg = json.dumps(arg_dict, separators=(',', ':'))
            user_args.append(arg)
        return user_args

    @staticmethod
    def _pdsh_include_to_nodelist(include_string: str):
        """If an `--include` string of the form `node1@node2` has been passed in, transforms it to a format SLURM will accept."""
        NODE_SEP = '@'
        SLOT_LIST_START = ':'
        if NODE_SEP not in include_string:
            return include_string
        if SLOT_LIST_START in include_string:
            raise NotImplementedError('Currently only allocating whole nodes is supported while using the SLURM launcher.')
        return include_string.replace(NODE_SEP, ',')
    @property
    def name(self):
        return 'slurm'

    def get_cmd(self, environment, active_resources):
        assert not getattr(self.args, 'detect_nvlink_pairs',
                           False), "slurm backend does not support remapping visible devices"
        total_process_count = sum(self.resource_pool.values())
        srun_cmd = [
            'srun',
            '-n',
            f'{total_process_count}',
        ] + split(self.args.launcher_args)

        if getattr(self.args, 'comment', ''):
            srun_cmd += ['--comment', self.args.comment]

        if getattr(self.args, 'account', ''):
            srun_cmd += ['--account', self.args.account]

        if self.args.include != "":
            srun_cmd.append('--nodelist')
            srun_cmd.append(self._pdsh_include_to_nodelist(self.args.include))

        if self.args.num_nodes > 0:
            srun_cmd.append('--nodes')
            srun_cmd.append(f'{self.args.num_nodes}')
        if self.args.num_gpus > 0:
            srun_cmd.append('--gpus')
            srun_cmd.append(f'{self.args.num_gpus}')

        exports = '--export=ALL'
        for key, val in self.exports.items():
            exports += f",{key}={val}"


    # run -n 2 --gpus 2 --mpi=pmix \
    # --export=ALL,NCCL_VERSION="2.25.1",NCCL_SOCKET_IFNAME="hsn",NCCL_NET_GDR_LEVEL="PHB",NCCL_HOME="/tools/brics/apps/nccl/v2.25.1-1-v1.6.x-r2/",NCCL_NET="AWS Libfabric",WANDB_API_KEY="689a18ccf36252478f4839de76b626c27f58bdea",PYTHONIOENCODING="utf-8",NCCL_CROSS_NIC="0",PYTHONPATH="/home/a5k/kyleobrien.a5k/gpt-neox" \
    # singularity exec \
        # --bind /home/a5k/kyleobrien.a5k/logs:/logs \
        # --bind /home/a5k/kyleobrien.a5k/data:/data \
        # --bind /home/a5k/kyleobrien.a5k/checkpoints:/checkpoints \
        # --bind /var/spool/slurmd/conf-cache:/var/spool/slurmd/conf-cache:ro \
        # --bind /home/a5k/kyleobrien.a5k/filtering_for_danger/lm_eval_tasks:/workspace/lm_eval_tasks \
        # --bind /home/a5k/kyleobrien.a5k/filtering_for_danger/neox/configs:/workspace/gpt-neox/configs/synced \
        # --bind /home/a5k/kyleobrien.a5k/gpt-neox:/workspace/local_repos/gpt-neox \
        # --bind /home/a5k/kyleobrien.a5k:/workspace/local_repos \
        # --bind /etc/slurm:/etc/slurm:ro \
        # --bind /var/run/munge:/var/run/munge \
        # --bind /usr/bin/srun:/usr/bin/srun:ro \
        # --bind /usr/bin/scontrol:/usr/bin/scontrol:ro \
        # --bind /usr/bin/sinfo:/usr/bin/sinfo:ro \
        # --bind /usr/bin/sbatch:/usr/bin/sbatch:ro \
        # --bind /usr/bin/scancel:/usr/bin/scancel:ro \
        # --bind /usr/bin/squeue:/usr/bin/squeue:ro \
        # --bind /usr/lib64/libslurm.so:/usr/lib64/libslurm.so:ro \
        # --bind /usr/lib64/libslurm.so.39:/usr/lib64/libslurm.so.39:ro \
        # --bind /usr/lib64/libslurm.so.39.0.0:/usr/lib64/libslurm.so.39.0.0:ro \
        # --bind /usr/lib64/slurm:/usr/lib64/slurm:ro \
        # --pwd /home/a5k/kyleobrien.a5k/gpt-neox \
        # --nv /home/a5k/kyleobrien.a5k/filtering_for_danger/neox/training-env.sif \
    #     bash -c "
    #     source /host/adapt.sh
    #     python -u train.py \
    #     --deepspeed_config eyJ0cmFpbl9iYXRjaF9zaXplIjogMzIsICJ0cmFpbl9taWNyb19iYXRjaF9zaXplX3Blcl9ncHUiOiAzMiwgIm9wdGltaXplciI6IHsidHlwZSI6ICJBZGFtIiwgInBhcmFtcyI6IHsibHIiOiAwLjAwMDMsICJiZXRhcyI6IFswLjksIDAuOTVdLCAiZXBzIjogMWUtMDh9fSwgInplcm9fb3B0aW1pemF0aW9uIjogeyJzdGFnZSI6IDEsICJhbGxnYXRoZXJfcGFydGl0aW9ucyI6IHRydWUsICJhbGxnYXRoZXJfYnVja2V0X3NpemUiOiAxMjYwMDAwMDAwLCAib3ZlcmxhcF9jb21tIjogdHJ1ZSwgInJlZHVjZV9zY2F0dGVyIjogdHJ1ZSwgInJlZHVjZV9idWNrZXRfc2l6ZSI6IDEyNjAwMDAwMDAsICJjb250aWd1b3VzX2dyYWRpZW50cyI6IHRydWUsICJjcHVfb2ZmbG9hZCI6IGZhbHNlfSwgIndhbGxfY2xvY2tfYnJlYWtkb3duIjogdHJ1ZSwgImJmMTYiOiB7ImVuYWJsZWQiOiB0cnVlfX0= \
    #     --megatron_config eyJudW1fZ3B1cyI6IDEsICJsYXVuY2hlciI6ICJzbHVybSIsICJub19zc2hfY2hlY2siOiB0cnVlLCAidHJhaW5fYmF0Y2hfc2l6ZSI6IDMyLCAidHJhaW5fbWljcm9fYmF0Y2hfc2l6ZV9wZXJfZ3B1IjogMzIsICJvcHRpbWl6ZXIiOiB7InR5cGUiOiAiQWRhbSIsICJwYXJhbXMiOiB7ImxyIjogMC4wMDAzLCAiYmV0YXMiOiBbMC45LCAwLjk1XSwgImVwcyI6IDFlLTA4fX0sICJ6ZXJvX29wdGltaXphdGlvbiI6IHsic3RhZ2UiOiAxLCAiYWxsZ2F0aGVyX3BhcnRpdGlvbnMiOiB0cnVlLCAiYWxsZ2F0aGVyX2J1Y2tldF9zaXplIjogMTI2MDAwMDAwMCwgIm92ZXJsYXBfY29tbSI6IHRydWUsICJyZWR1Y2Vfc2NhdHRlciI6IHRydWUsICJyZWR1Y2VfYnVja2V0X3NpemUiOiAxMjYwMDAwMDAwLCAiY29udGlndW91c19ncmFkaWVudHMiOiB0cnVlLCAiY3B1X29mZmxvYWQiOiBmYWxzZX0sICJ3YWxsX2Nsb2NrX2JyZWFrZG93biI6IHRydWUsICJkZWVwc3BlZWRfZXh0cmFfYXJncyI6IHsiYmYxNiI6IHsiZW5hYmxlZCI6IHRydWV9fSwgInByZWNpc2lvbiI6ICJiZmxvYXQxNiIsICJudW1fbGF5ZXJzIjogMzIsICJoaWRkZW5fc2l6ZSI6IDQwOTYsICJudW1fYXR0ZW50aW9uX2hlYWRzIjogMzIsICJzZXFfbGVuZ3RoIjogMjA0OCwgIm1heF9wb3NpdGlvbl9lbWJlZGRpbmdzIjogMjA0OCwgInBvc19lbWIiOiAicm90YXJ5IiwgIm5vX3dlaWdodF90eWluZyI6IHRydWUsICJhdHRlbnRpb25fY29uZmlnIjogWyJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCIsICJmbGFzaCJdLCAic3BhcnNpdHlfY29uZmlnIjoge30sICJzY2FsZWRfdXBwZXJfdHJpYW5nX21hc2tlZF9zb2Z0bWF4X2Z1c2lvbiI6IHRydWUsICJyb3RhcnlfcGN0IjogMC4yNSwgImdwdF9qX3Jlc2lkdWFsIjogdHJ1ZSwgInRlX2xheWVybm9ybV9tbHAiOiB0cnVlLCAidGVfbWhhIjogdHJ1ZSwgInRlX2ZwOF93Z3JhZCI6IGZhbHNlLCAibHJfZGVjYXlfc3R5bGUiOiAiY29zaW5lIiwgImxyX2RlY2F5X2l0ZXJzIjogMzgxNDY5NywgIm1pbl9sciI6IDEuMmUtMDUsICJvcHRpbWl6ZXJfdHlwZSI6ICJBZGFtIiwgInplcm9fc3RhZ2UiOiAxLCAiemVyb19yZWR1Y2Vfc2NhdHRlciI6IHRydWUsICJ6ZXJvX2NvbnRpZ3VvdXNfZ3JhZGllbnRzIjogdHJ1ZSwgInplcm9fcmVkdWNlX2J1Y2tldF9zaXplIjogMTI2MDAwMDAwMCwgInplcm9fYWxsZ2F0aGVyX2J1Y2tldF9zaXplIjogMTI2MDAwMDAwMCwgImxyIjogMC4wMDAzLCAidG9rZW5pemVyX3R5cGUiOiAiSEZUb2tlbml6ZXIiLCAiZGF0YV9wYXRoIjogIi9kYXRhL2Vud2lrOC9lbndpazhfdGV4dF9kb2N1bWVudCIsICJkYXRhX2ltcGwiOiAibW1hcCIsICJzYXZlIjogIi9jaGVja3BvaW50cy9pc2FtYmFyZF9zaW5nbGVfbm9kZV90ZXN0X3ByZXRyYWluaW5nIiwgImNvbmZpZ19maWxlcyI6IHsiaXNhbWJhcmRfc2luZ2xlX25vZGVfdGVzdC55bWwiOiAie1xuICAjIFRva2Vuc1xuICBcImRhdGFfcGF0aFwiOiBcIi9kYXRhL2Vud2lrOC9lbndpazhfdGV4dF9kb2N1bWVudFwiLFxuICBcInZvY2FiX2ZpbGVcIjogXCIvZGF0YS9uZW94X3Rva2VuaXplci90b2tlbml6ZXIuanNvblwiLFxuICBcInRva2VuaXplcl90eXBlXCI6IFwiSEZUb2tlbml6ZXJcIixcbiAgXCJkYXRhX2ltcGxcIjogXCJtbWFwXCIsXG5cbiAgIyBMb2dnaW5nXG4gIFwiY2hlY2twb2ludF92YWxpZGF0aW9uX3dpdGhfZm9yd2FyZF9wYXNzXCI6IGZhbHNlLFxuICBcInRlbnNvcmJvYXJkX2RpclwiOiBcInRlbnNvcmJvYXJkXCIsXG4gIFwibG9nX2RpclwiOiBcIi9sb2dzXCIsXG4gIFwibG9nX2ludGVydmFsXCI6IDEwLFxuICBcInN0ZXBzX3Blcl9wcmludFwiOiAxMCxcbiAgXCJ3YWxsX2Nsb2NrX2JyZWFrZG93blwiOiB0cnVlLFxuICBcInVzZV93YW5kYlwiOiB0cnVlLFxuICBcIndhbmRiX2hvc3RcIjogXCJodHRwczovL2FwaS53YW5kYi5haVwiLFxuICBcIndhbmRiX3Byb2plY3RcIjogXCJBSVNJXCIsXG4gIFwid2FuZGJfdGVhbVwiOiBcImVsZXV0aGVyYWlcIixcbiAgXCJ3YW5kYl9ydW5fbmFtZVwiOiBcImlzYW1iYXJkX3NpbmdsZV9ub2RlX3Rlc3RcIixcblxuICAjIERpc3RyaWJ1dGVkIFRyYWluaW5nIC0gRm9yIGxvY2FsIGV4ZWN1dGlvblxuICAjIFJlbW92ZSBhbGwgbGF1bmNoZXItcmVsYXRlZCBzZXR0aW5ncyBzaW5jZSB3ZSdyZSBydW5uaW5nIGxvY2FsbHlcbiAgXCJwaXBlX3BhcmFsbGVsX3NpemVcIjogMSxcbiAgXCJtb2RlbF9wYXJhbGxlbF9zaXplXCI6IDEsXG5cbiAgIyBUcmFpbmluZyBEdXJhdGlvbiAtIEFkanVzdGVkIGZvciBzaW5nbGUgbm9kZVxuICAjIDUwMEIgKHRva2VucykgLyAoMSAoZ3JhZCBhY2MpICogNCAoR1BVcykgKiAzMiAobWljcm8gYmF0Y2ggc2l6ZSkgKiAyMDQ4IChzZXEgbGVuZ3RoKSlcbiAgXCJ0cmFpbl9pdGVyc1wiOiAzODE0Njk3LFxuICBcImxyX2RlY2F5X2l0ZXJzXCI6IDM4MTQ2OTcsXG4gIFwiZGlzdHJpYnV0ZWRfYmFja2VuZFwiOiBcIm5jY2xcIixcbiAgXCJscl9kZWNheV9zdHlsZVwiOiBcImNvc2luZVwiLFxuICBcIndhcm11cFwiOiAwLjAxLFxuICBcInNwbGl0XCI6IFwiMTAwLDAsMFwiLFxuXG4gIFwibGF1bmNoZXJcIjogXCJzbHVybVwiLFxuICBcImRlZXBzcGVlZF9zbHVybVwiOiB0cnVlLFxuICBcIm5vX3NzaF9jaGVja1wiOiB0cnVlLFxuICBcIm51bV9ncHVzXCI6IDEsXG5cbiAgIyBBcmNoaXRlY3R1cmVcbiAgXCJudW1fbGF5ZXJzXCI6IDMyLFxuICBcImhpZGRlbl9zaXplXCI6IDQwOTYsXG4gIFwibnVtX2F0dGVudGlvbl9oZWFkc1wiOiAzMixcbiAgXCJzZXFfbGVuZ3RoXCI6IDIwNDgsXG4gIFwibWF4X3Bvc2l0aW9uX2VtYmVkZGluZ3NcIjogMjA0OCxcbiAgXCJub3JtXCI6IFwibGF5ZXJub3JtXCIsXG4gIFwicG9zX2VtYlwiOiBcInJvdGFyeVwiLFxuICBcInJvdGFyeV9wY3RcIjogMC4yNSxcbiAgXCJub193ZWlnaHRfdHlpbmdcIjogdHJ1ZSxcbiAgXCJncHRfal9yZXNpZHVhbFwiOiB0cnVlLFxuICBcIm91dHB1dF9sYXllcl9wYXJhbGxlbGlzbVwiOiBcImNvbHVtblwiLFxuICBcImF0dGVudGlvbl9jb25maWdcIjogW1tbXCJmbGFzaFwiXSwgMzJdXSxcbiAgXCJzY2FsZWRfdXBwZXJfdHJpYW5nX21hc2tlZF9zb2Z0bWF4X2Z1c2lvblwiOiB0cnVlLFxuICBcInByZWNpc2lvblwiOiBcImJmbG9hdDE2XCIsXG4gIFwiYWN0aXZhdGlvblwiOiBcImdlbHVcIixcblxuICAjIFRyYW5zZm9ybWVyIEVuZ2luZVxuICBcInRlX2NvbHVtbnBhcmFsbGVsXCI6IGZhbHNlLFxuICBcInRlX3Jvd3BhcmFsbGVsXCI6IGZhbHNlLFxuICBcInRlX2xheWVybm9ybV9tbHBcIjogdHJ1ZSxcbiAgXCJ0ZV9taGFcIjogdHJ1ZSxcbiAgXCJ0ZV9mcDhfZm9ybWF0XCI6IFwiaHlicmlkXCIsXG4gIFwidGVfZnA4X3dncmFkXCI6IGZhbHNlLFxuICBcInRlX2ZwOF9hbWF4X2hpc3RvcnlfbGVuXCI6IDEsXG4gIFwidGVfZnA4X2FtYXhfY29tcHV0ZV9hbGdvXCI6IFwibW9zdF9yZWNlbnRcIixcbiAgXCJ0ZV9mcDhfbWFyZ2luXCI6IDAsXG4gIFwidGVfZnA4X21oYVwiOiBmYWxzZSxcblxuICAjIE9wdGltaXphdGlvblxuICBcIm9wdGltaXplclwiOiB7XG4gICAgXCJ0eXBlXCI6IFwiQWRhbVwiLFxuICAgIFwicGFyYW1zXCI6IHsgXCJsclwiOiAwLjAwMDMsIFwiYmV0YXNcIjogWzAuOSwgMC45NV0sIFwiZXBzXCI6IDEuMGUtOCB9XG4gIH0sXG4gIFwibWluX2xyXCI6IDAuMDAwMDEyLFxuICBcInplcm9fb3B0aW1pemF0aW9uXCI6IHtcbiAgICBcInN0YWdlXCI6IDEsXG4gICAgXCJhbGxnYXRoZXJfcGFydGl0aW9uc1wiOiB0cnVlLFxuICAgIFwiYWxsZ2F0aGVyX2J1Y2tldF9zaXplXCI6IDEyNjAwMDAwMDAsXG4gICAgXCJvdmVybGFwX2NvbW1cIjogdHJ1ZSxcbiAgICBcInJlZHVjZV9zY2F0dGVyXCI6IHRydWUsXG4gICAgXCJyZWR1Y2VfYnVja2V0X3NpemVcIjogMTI2MDAwMDAwMCxcbiAgICBcImNvbnRpZ3VvdXNfZ3JhZGllbnRzXCI6IHRydWUsXG4gICAgXCJjcHVfb2ZmbG9hZFwiOiBmYWxzZVxuICB9LFxuICBcInRyYWluX21pY3JvX2JhdGNoX3NpemVfcGVyX2dwdVwiOiAzMixcbiAgXCJncmFkaWVudF9hY2N1bXVsYXRpb25fc3RlcHNcIjogMSxcbiAgXCJncmFkaWVudF9jbGlwcGluZ1wiOiAxLjAsXG4gIFwid2VpZ2h0X2RlY2F5XCI6IDAuMSxcbiAgXCJoaWRkZW5fZHJvcG91dFwiOiAwLFxuICBcImF0dGVudGlvbl9kcm9wb3V0XCI6IDAsXG5cbiAgIyBDaGVja3BvaW50aW5nXG4gIFwiY2hlY2twb2ludF9hY3RpdmF0aW9uc1wiOiB0cnVlLFxuICBcImNoZWNrcG9pbnRfbnVtX2xheWVyc1wiOiAxLFxuICBcInBhcnRpdGlvbl9hY3RpdmF0aW9uc1wiOiB0cnVlLFxuICBcInN5bmNocm9uaXplX2VhY2hfbGF5ZXJcIjogdHJ1ZSxcbiAgXCJjaGVja3BvaW50X2ZhY3RvclwiOiAxMTkyLFxuICBcInNhdmVcIjogXCIvY2hlY2twb2ludHMvaXNhbWJhcmRfc2luZ2xlX25vZGVfdGVzdF9wcmV0cmFpbmluZ1wiLFxuICBcImxvYWRcIjogXCIvY2hlY2twb2ludHMvaXNhbWJhcmRfc2luZ2xlX25vZGVfdGVzdF9wcmV0cmFpbmluZ1wiLFxuXG4gICMgRXZhbHVhdGlvblxuICBcImV2YWxfaXRlcnNcIjogMCxcbiAgXCJldmFsX2ludGVydmFsXCI6IDUsXG4gIFwiZXZhbF9yZXN1bHRzX3ByZWZpeFwiOiBcImlzYW1iYXJkX3NpbmdsZV9ub2RlX3Rlc3RcIlxufSJ9LCAibG9hZCI6ICIvY2hlY2twb2ludHMvaXNhbWJhcmRfc2luZ2xlX25vZGVfdGVzdF9wcmV0cmFpbmluZyIsICJjaGVja3BvaW50X2ZhY3RvciI6IDExOTIsICJiYXRjaF9zaXplIjogMzIsICJ0cmFpbl9pdGVycyI6IDM4MTQ2OTcsICJldmFsX2l0ZXJzIjogMCwgImV2YWxfaW50ZXJ2YWwiOiA1LCAic3BsaXQiOiAiMTAwLDAsMCIsICJ2b2NhYl9maWxlIjogIi9kYXRhL25lb3hfdG9rZW5pemVyL3Rva2VuaXplci5qc29uIiwgImNoZWNrcG9pbnRfYWN0aXZhdGlvbnMiOiB0cnVlLCAic3luY2hyb25pemVfZWFjaF9sYXllciI6IHRydWUsICJwYXJ0aXRpb25fYWN0aXZhdGlvbnMiOiB0cnVlLCAiZHluYW1pY19sb3NzX3NjYWxlIjogdHJ1ZSwgInBpcGVfcGFyYWxsZWxfc2l6ZSI6IDEsICJ3b3JsZF9zaXplIjogMSwgImlzX3BpcGVfcGFyYWxsZWwiOiB0cnVlLCAidXNlX3dhbmRiIjogdHJ1ZSwgIndhbmRiX2dyb3VwIjogIml3YzYzYWVqX3NrN2xmaWpiIiwgIndhbmRiX3J1bl9uYW1lIjogImlzYW1iYXJkX3NpbmdsZV9ub2RlX3Rlc3QiLCAid2FuZGJfdGVhbSI6ICJlbGV1dGhlcmFpIiwgIndhbmRiX3Byb2plY3QiOiAiQUlTSSIsICJsb2dfZGlyIjogIi9sb2dzIiwgInRlbnNvcmJvYXJkX2RpciI6ICJ0ZW5zb3Jib2FyZCIsICJsb2dfaW50ZXJ2YWwiOiAxMCwgInRleHRfZ2VuX3R5cGUiOiAidW5jb25kaXRpb25hbCIsICJldmFsX3Jlc3VsdHNfcHJlZml4IjogImlzYW1iYXJkX3NpbmdsZV9ub2RlX3Rlc3QiLCAibG9jYWxfcmFuayI6IDAsICJyYW5rIjogMCwgImRlZXBzcGVlZF9zbHVybSI6IHRydWUsICJ1c2VyX3NjcmlwdCI6ICJ0cmFpbi5weSIsICJnbG9iYWxfbnVtX2dwdXMiOiAxfQ==
    #     "
        python_exec = [sys.executable, "-u"]
        singularity_args = [
            "singularity exec",
            "--bind /home/a5k/kyleobrien.a5k/logs:/logs",
            "--bind /home/a5k/kyleobrien.a5k/data:/data",
            "--bind /home/a5k/kyleobrien.a5k/checkpoints:/checkpoints",
            "--bind /var/spool/slurmd/conf-cache:/var/spool/slurmd/conf-cache:ro",
            "--bind /home/a5k/kyleobrien.a5k/filtering_for_danger/lm_eval_tasks:/workspace/lm_eval_tasks",
            "--bind /home/a5k/kyleobrien.a5k/filtering_for_danger/neox/configs:/workspace/gpt-neox/configs/synced",
            "--bind /home/a5k/kyleobrien.a5k/gpt-neox:/workspace/local_repos/gpt-neox",
            "--bind /home/a5k/kyleobrien.a5k:/workspace/local_repos",
            "--bind /etc/slurm:/etc/slurm:ro",
            "--bind /var/run/munge:/var/run/munge",
            "--bind /usr/bin/srun:/usr/bin/srun:ro",
            "--bind /usr/bin/scontrol:/usr/bin/scontrol:ro",
            "--bind /usr/bin/sinfo:/usr/bin/sinfo:ro",
            "--bind /usr/bin/sbatch:/usr/bin/sbatch:ro",
            "--bind /usr/bin/scancel:/usr/bin/scancel:ro",
            "--bind /usr/bin/squeue:/usr/bin/squeue:ro",
            "--bind /usr/lib64/libslurm.so:/usr/lib64/libslurm.so:ro",
            "--bind /usr/lib64/libslurm.so.39:/usr/lib64/libslurm.so.39:ro",
            "--bind /usr/lib64/libslurm.so.39.0.0:/usr/lib64/libslurm.so.39.0.0:ro",
            "--bind /usr/lib64/slurm:/usr/lib64/slurm:ro",
            "--pwd /home/a5k/kyleobrien.a5k/gpt-neox",
            "--nv",
            "/home/a5k/kyleobrien.a5k/filtering_for_danger/neox/training-env.sif",
        ]
        launch_script = f"""bash -c "
        source /host/adapt.sh
        {python_exec} {self.user_script} {self.user_arguments}
        """

        # command = srun_cmd + [exports] + python_exec + [self.user_script] + self.user_arguments
        command_with_singularity = srun_cmd + [exports] + singularity_args + [launch_script]
        return command_with_singularity


class MVAPICHRunner(MultiNodeRunner):

    def __init__(self, args, world_info_base64, resource_pool):
        super().__init__(args, world_info_base64)
        self.resource_pool = resource_pool

        # Disable the CMA kernel module, not available on Ubuntu systems
        self.add_export('MV2_SMP_USE_CMA', '0')

        # If we fail this will output more verbose logging
        self.add_export('MV2_DEBUG_SHOW_BACKTRACE', '1')

        # Enabled cuda-aware communication
        if get_accelerator().device_name() == 'cuda':
            self.add_export('MV2_USE_CUDA', '1')

        # Support deep learning frameworks: http://hidl.cse.ohio-state.edu/userguide/horovod/
        self.add_export('MV2_SUPPORT_DL', '1')

        # Support MPI_THREAD_MULTIPLE
        self.add_export('MV2_ENABLE_AFFINITY', '0')

        # Performance tuning flags for allgather
        self.add_export('MV2_INTER_ALLGATHER_TUNING', '5')
        self.add_export('MV2_CUDA_USE_NAIVE', '0')

    def backend_exists(self):
        #TODO: if IB is available we should suggestion mvapich
        mpiname_exists = shutil.which('mpiname')
        exists = False
        if not mpiname_exists:
            warnings.warn("mpiname does not exist, mvapich is not installed properly")
        else:
            results = subprocess.check_output(['mpiname'])
            mpiname_results = results.decode('utf-8').strip()
            if "MVAPICH2-GDR" in mpiname_results:
                exists = True
            else:
                warnings.warn(f"Expected MVAPICH2-GDR as return for mpiname but received {mpiname_results}")
        return exists

    @property
    def name(self):
        return "mvapich"

    def validate_args(self):
        super().validate_args()
        #TODO: Allow for include/exclude at node-level but not gpu-level
        if self.args.include != "" or self.args.exclude != "":
            raise ValueError(f"{self.name} backend does not support worker include/exclusion")
        if self.args.num_nodes != -1 or self.args.num_gpus != -1:
            raise ValueError(f"{self.name} backend does not support limiting num nodes/gpus")

    def get_cmd(self, environment, active_resources):
        devices_per_node = self.resource_pool.values()
        total_process_count = sum(devices_per_node)
        process_per_node = list(devices_per_node)[0]
        if not all([n == process_per_node for n in devices_per_node]):
            raise ValueError("mvapich requires same number of devices per node")

        with open(MVAPICH_TMP_HOSTFILE, 'w') as fd:
            for host in self.resource_pool.keys():
                fd.write(f'{host}\n')

        mpirun_cmd = [
            'mpirun',
            '-np',
            f'{total_process_count}',
            '-ppn',
            f'{process_per_node}',
            '--hostfile',
            f'{MVAPICH_TMP_HOSTFILE}',
        ] + split(self.args.launcher_args)

        export_cmd = []
        for k, v in self.exports.items():
            export_cmd += ['-env', "{}={}".format(k, v)]

        python_exec = []
        if not self.args.no_python:
            python_exec = [sys.executable, "-u"]
            if self.args.module:
                python_exec.append("-m")

        return mpirun_cmd + export_cmd + python_exec + [self.user_script] + self.user_arguments
