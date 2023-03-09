<<<<<<< HEAD
"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from deepspeed.runtime.config_utils import get_scalar_param, DeepSpeedConfigObject
from deepspeed.utils import logger
from .constants import *
from .offload_constants import *
from .offload_config import get_offload_param_config, get_default_offload_param_config, \
    get_offload_optimizer_config, get_default_offload_optimizer_config


class DeepSpeedZeroConfig(DeepSpeedConfigObject):
    def __init__(self, param_dict):
        super(DeepSpeedZeroConfig, self).__init__()

        self.stage = None
        self.contiguous_gradients = None
        self.reduce_scatter = None
        self.reduce_bucket_size = None
        self.allgather_partitions = None
        self.allgather_bucket_size = None
        self.overlap_comm = None
        self.load_from_fp32_weights = None

        self.elastic_checkpoint = None

        #Offload Specific Parameters
        self.offload_param = None
        self.offload_optimizer = None
        self.sub_group_size = None

        #Stage3 Specific Parameters
        self.prefetch_bucket_size = None
        self.param_persistence_threshold = None
        self.max_live_parameters = None
        self.max_reuse_distance = None
        self.gather_fp16_weights_on_model_save = None

        if ZERO_OPTIMIZATION in param_dict.keys():
            zero_config_dict = param_dict[ZERO_OPTIMIZATION]
            if type(zero_config_dict) is bool:
                zero_config_dict = self.read_zero_config_deprecated(param_dict)
        else:
            zero_config_dict = ZERO_OPTIMIZATION_DEFAULT

        self._initialize(zero_config_dict)

    def read_zero_config_deprecated(self, param_dict):
        zero_config_dict = {}
        zero_config_dict[
            ZERO_OPTIMIZATION_STAGE] = 1 if param_dict[ZERO_OPTIMIZATION] else 0
        if zero_config_dict[ZERO_OPTIMIZATION_STAGE] > 0:
            zero_config_dict[ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE] = get_scalar_param(
                param_dict,
                ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE_DEPRECATED,
                ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE_DEFAULT)

        logger.warning(
            'DeepSpeedConfig: this format of ZeRO optimization setup is deprecated. Please use the following format: {}'
            .format(ZERO_FORMAT))
        return zero_config_dict

    def _sanity_check(self, zero_config_dict):
        deprecated_dict = {
            ZERO_OPTIMIZATION_CPU_OFFLOAD:
            ZERO_OPTIMIZATION_OFFLOAD_OPTIMIZER,
            ZERO_OPTIMIZATION_CPU_OFFLOAD_PARAMS:
            ZERO_OPTIMIZATION_OFFLOAD_PARAM,
            ZERO_OPTIMIZATION_CPU_OFFLOAD_USE_PIN_MEMORY:
            f'{ZERO_OPTIMIZATION_OFFLOAD_PARAM} or {ZERO_OPTIMIZATION_OFFLOAD_OPTIMIZER}'
        }

        for old_key, new_key in deprecated_dict.items():
            if old_key in zero_config_dict:
                logger.warning(
                    f'DeepSpeedConfig: {old_key} is deprecated. Please use {new_key}.')

    def _initialize(self, zero_config_dict):
        self._sanity_check(zero_config_dict)

        self.stage = get_scalar_param(zero_config_dict,
                                      ZERO_OPTIMIZATION_STAGE,
                                      ZERO_OPTIMIZATION_STAGE_DEFAULT)

        self.contiguous_gradients = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_CONTIGUOUS_GRADIENTS,
            ZERO3_OPTIMIZATION_CONTIGUOUS_GRADIENTS_DEFAULT
            if self.stage == ZERO_OPTIMIZATION_WEIGHTS else
            ZERO_OPTIMIZATION_CONTIGUOUS_GRADIENTS_DEFAULT)

        self.reduce_bucket_size = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE,
            ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE_DEFAULT)

        self.reduce_scatter = get_scalar_param(zero_config_dict,
                                               ZERO_OPTIMIZATION_REDUCE_SCATTER,
                                               ZERO_OPTIMIZATION_REDUCE_SCATTER_DEFAULT)

        self.overlap_comm = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_OVERLAP_COMM,
            ZERO3_OPTIMIZATION_OVERLAP_COMM_DEFAULT
            if self.stage == ZERO_OPTIMIZATION_WEIGHTS else
            ZERO_OPTIMIZATION_OVERLAP_COMM_DEFAULT)

        self.allgather_partitions = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_ALLGATHER_PARTITIONS,
            ZERO_OPTIMIZATION_ALLGATHER_PARTITIONS_DEFAULT)

        self.allgather_bucket_size = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE,
            ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE_DEFAULT)

        self.load_from_fp32_weights = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_LOAD_FROM_FP32_WEIGHTS,
            ZERO_OPTIMIZATION_LOAD_FROM_FP32_WEIGHTS_DEFAULT)

        self.elastic_checkpoint = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_ELASTIC_CHECKPOINT,
            ZERO_OPTIMIZATION_ELASTIC_CHECKPOINT_DEFAULT)

        if ZERO_OPTIMIZATION_CPU_OFFLOAD in zero_config_dict:
            cpu_offload_optimizer = get_scalar_param(
                zero_config_dict,
                ZERO_OPTIMIZATION_CPU_OFFLOAD,
                ZERO_OPTIMIZATION_CPU_OFFLOAD_DEFAULT)
            if cpu_offload_optimizer:
                self.offload_optimizer = get_default_offload_optimizer_config()
        else:
            self.offload_optimizer = get_offload_optimizer_config(zero_config_dict)

        if ZERO_OPTIMIZATION_CPU_OFFLOAD_PARAMS in zero_config_dict:
            cpu_offload_params = get_scalar_param(
                zero_config_dict,
                ZERO_OPTIMIZATION_CPU_OFFLOAD_PARAMS,
                ZERO_OPTIMIZATION_CPU_OFFLOAD_PARAMS_DEFAULT)
            if cpu_offload_params:
                self.offload_param = get_default_offload_param_config()
        else:
            self.offload_param = get_offload_param_config(zero_config_dict)

        self.sub_group_size = get_scalar_param(zero_config_dict,
                                               ZERO_OPTIMIZATION_SUB_GROUP_SIZE,
                                               ZERO_OPTIMIZATION_SUB_GROUP_SIZE_DEFAULT)

        self.max_live_parameters = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_MAX_LIVE_PARAMETERS,
            ZERO_OPTIMIZATION_MAX_LIVE_PARAMETERS_DEFAULT)

        self.max_reuse_distance = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_MAX_REUSE_DISTANCE,
            ZERO_OPTIMIZATION_MAX_REUSE_DISTANCE_DEFAULT)

        self.prefetch_bucket_size = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_PREFETCH_BUCKET_SIZE,
            ZERO_OPTIMIZATION_PREFETCH_BUCKET_SIZE_DEFAULT)

        self.param_persistence_threshold = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_PARAM_PERSISTENCE_THRESHOLD,
            ZERO_OPTIMIZATION_PARAM_PERSISTENCE_THRESHOLD_DEFAULT)

        self.gather_fp16_weights_on_model_save = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_GATHER_FP16_WEIGHTS_ON_MODEL_SAVE,
            ZERO_OPTIMIZATION_GATHER_FP16_WEIGHTS_ON_MODEL_SAVE_DEFAULT)
=======
'''Copyright The Microsoft DeepSpeed Team'''
"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from pydantic import Field, validator
import sys
from typing import Optional
from enum import Enum
from deepspeed.runtime.config_utils import get_scalar_param, pp_int, DeepSpeedConfigModel
from deepspeed.utils import logger
from .offload_config import DeepSpeedZeroOffloadParamConfig, DeepSpeedZeroOffloadOptimizerConfig, OffloadDeviceEnum

# ZeRO optimization. By default, this optimization is not enabled.
# Users have to configure the desired optimization (0 means disabled) in params.json as below example:
ZERO_FORMAT = """
ZeRO optimization should be enabled as:
"session_params": {
  "zero_optimization": {
    "stage": [0|1|2],
    "stage3_max_live_parameters" : 1000000000,
    "stage3_max_reuse_distance" : 1000000000,
    "allgather_partitions": [true|false],
    "allgather_bucket_size": 500000000,
    "reduce_scatter": [true|false],
    "contiguous_gradients" : [true|false]
    "overlap_comm": [true|false],
    "reduce_bucket_size": 500000000,
    "load_from_fp32_weights": [true|false],
    "cpu_offload": [true|false] (deprecated),
    "cpu_offload_params" : [true|false] (deprecated),
    "cpu_offload_use_pin_memory": [true|false] (deprecated),
    "sub_group_size" : 1000000000000,
    "offload_param": {...},
    "offload_optimizer": {...},
    "ignore_unused_parameters": [true|false],
    "round_robin_gradients": [true|false]
    }
}
"""

ZERO_OPTIMIZATION = "zero_optimization"


def read_zero_config_deprecated(param_dict):
    zero_config_dict = {}
    zero_config_dict["stage"] = 1 if param_dict[ZERO_OPTIMIZATION] else 0
    if zero_config_dict["stage"] > 0:
        zero_config_dict["allgather_bucket_size"] = get_scalar_param(
            param_dict,
            "allgather_size",
            5e8)
    logger.warning(
        "DeepSpeedConfig: this format of ZeRO optimization setup is deprecated. Please use the following format: {}"
        .format(ZERO_FORMAT))
    return zero_config_dict


def get_zero_config(param_dict):
    if ZERO_OPTIMIZATION in param_dict:
        zero_config_dict = param_dict[ZERO_OPTIMIZATION]
        if isinstance(zero_config_dict, bool):
            zero_config_dict = read_zero_config_deprecated(param_dict)
    else:
        zero_config_dict = {}
    return DeepSpeedZeroConfig(**zero_config_dict)


class ZeroStageEnum(int, Enum):
    """ Enum class for possible zero stages """
    disabled = 0
    optimizer_states = 1
    gradients = 2
    weights = 3
    max_stage = 3


class DeepSpeedZeroConfig(DeepSpeedConfigModel):
    """
    Sets parameters for ZeRO optimizations.
    """

    stage: ZeroStageEnum = 0
    """
    Chooses different stages of ZeRO Optimizer. Stage 0, 1, 2, and 3 refer
    to disabled, optimizer state partitioning, and optimizer+gradient state
    partitioning, and optimizer+gradient+parameter partitioning, respectively.
    """

    contiguous_gradients: bool = True
    """
    Copies the gradients to a contiguous buffer as they are produced. Avoids
    memory fragmentation during backward pass.
    """

    reduce_scatter: bool = True
    """
    Uses reduce or reduce scatter instead of allreduce to average gradients
    """

    reduce_bucket_size: int = Field(pp_int(5e8), ge=0)
    """
    Number of elements reduced/allreduced at a time. Limits the memory required
    for the allgather for large model sizes
    """

    allgather_partitions: bool = True
    """
    Chooses between allgather collective or a series of broadcast collectives
    to gather updated parameters from all the GPUs at the end of each step
    """

    allgather_bucket_size: int = Field(pp_int(5e8), ge=0)
    """
    Number of elements allgathered at a time. Limits the memory required for
    the allgather for large model sizes
    """

    overlap_comm: bool = None  # None for dynamic default value (see validator `overlap_comm_valid` below)
    """
    Attempts to overlap the reduction of the gradients with backward computation
    """

    load_from_fp32_weights: bool = True
    """
    Boolean indicating whether to initialize fp32 master weights from fp32
    copies in checkpoint (no precision loss) or from model's fp16 copies (with
    precision loss). This can be used to initialize optimizer state even when
    checkpoint is missing optimizer state.
    """

    elastic_checkpoint: bool = False
    """
    Enable loading checkpoint that was saved by job with different GPU count.
    No longer supported.
    """

    offload_param: Optional[DeepSpeedZeroOffloadParamConfig] = None
    """
    Enable offloading of model parameters to CPU or NVMe. This frees up GPU
    memory for larger models or batch sizes. Valid only with stage 3. Expects a
    dictionary containing values for :any:`DeepSpeedZeroOffloadParamConfig`.
    """

    offload_optimizer: Optional[DeepSpeedZeroOffloadOptimizerConfig] = None
    """
    Enable offloading of optimizer state to CPU or NVMe, and optimizer
    computation to CPU. This frees up GPU memory for larger models or batch
    sizes. Valid for ZeRO stage 1, 2, 3. Expects a dictionary containing values
    for :any:`DeepSpeedZeroOffloadOptimizerConfig`.
    """

    sub_group_size: int = Field(pp_int(1e9), ge=0)
    """
    Tile size for parameter processing to fit massive models (with trillions of
    parameters). Used by ZeRO3-Offload and ZeRO-Infinity
    """

    cpu_offload_param: bool = Field(
        None,
        deprecated=True,
        new_param="offload_param",
        new_param_fn=(
            lambda val: DeepSpeedZeroOffloadParamConfig(device=OffloadDeviceEnum.cpu)
            if val else None),
    )
    """ Deprecated, please use ``offload_param`` """

    cpu_offload_use_pin_memory: bool = Field(
        None,
        deprecated=True,
        new_param="offload_param or offload_optimizer",
        set_new_param=False,
    )
    """ Deprecated, please use ``offload_param`` or ``offload_optimizer`` """

    cpu_offload: bool = Field(
        None,
        deprecated=True,
        new_param="offload_optimizer",
        new_param_fn=(
            lambda val: DeepSpeedZeroOffloadOptimizerConfig(device=OffloadDeviceEnum.cpu)
            if val else None),
    )
    """ Deprecated, please use ``offload_optimizer`` """

    prefetch_bucket_size: int = Field(pp_int(5e7),
                                      ge=0,
                                      alias="stage3_prefetch_bucket_size")
    """
    Maximum number of parameter elements to fetch ahead of use. Used by ZeRO3,
    ZeRO3-Offload, ZeRO-Infinity, and ZeRO-Inference.
    """

    param_persistence_threshold: int = Field(pp_int(1e5),
                                             ge=0,
                                             alias="stage3_param_persistence_threshold")
    """
    Do not partition parameters smaller than this threshold. Smaller values use
    less memory, but can greatly increase communication (especially
    latency-bound messages).
    """

    model_persistence_threshold: int = Field(pp_int(sys.maxsize,
                                                    "sys.maxsize"),
                                             ge=0,
                                             alias="stage3_model_persistence_threshold")
    """
    Maximum number of parameter elements that can be persisted in GPU and not
    partitioned. This imposes an upper bound on the number of unpartitioned
    parameters resulting from param_persistence_threshold setting. Used by
    ZeRO3-Offload, ZeRO-Infinity and ZeRO-Inference.
    """

    max_live_parameters: int = Field(pp_int(1e9),
                                     ge=0,
                                     alias="stage3_max_live_parameters")
    """
    The maximum number of parameters resident per GPU before releasing. Smaller
    values use less memory, but perform more communication.
    """

    max_reuse_distance: int = Field(pp_int(1e9), ge=0, alias="stage3_max_reuse_distance")
    """
    Do not release a parameter if it will be reused within this threshold of
    parameters. Smaller values use less memory, but perform more communication.
    """

    gather_16bit_weights_on_model_save: bool = Field(
        False,
        alias="stage3_gather_16bit_weights_on_model_save")
    """
    Consolidate the weights before saving the model by ``save_16bit_model()``.
    Since the weights are partitioned across GPUs, they aren’t part of
    ``state_dict``, so this function automatically gathers the weights when
    this option is enabled and then saves the fp16 model weights.
    """

    stage3_gather_fp16_weights_on_model_save: bool = Field(
        False,
        deprecated=True,
        new_param="gather_16bit_weights_on_model_save")
    """ Deprecated, please use ``gather_16bit_weights_on_model_save`` """

    ignore_unused_parameters: bool = True
    """
    Unused parameters in modules may be unexpected in static networks, but
    could be normal in dynamic networks. This controls whether or not training
    should terminate with an error message when unused parameters are detected.
    This is set to ``False`` by default, which means unused parameters are
    ignored and training continues. Now is just used in stage 2.
    """

    legacy_stage1: bool = False
    """
    For backward-compatibility enable old ZeRO stage 1 implementation. Use at
    your own risk, will be deprecated soon.
    """

    round_robin_gradients: bool = False
    """
    Stage 1 and 2 optimization for CPU offloading that parallelizes gradient
    copying to CPU memory among ranks by fine-grained gradient partitioning.
    Performance benefit grows with gradient accumulation steps (more copying
    between optimizer steps) or GPU count (increased parallelism).
    """

    # Validators
    @validator("overlap_comm")
    def overlap_comm_valid(cls, field_value, values):
        if field_value is None:
            assert (
                "stage" in values
            ), "DeepSpeedZeroConfig: 'stage' must be defined before 'overlap_comm'"
            field_value = values["stage"] == ZeroStageEnum.weights
        return field_value
>>>>>>> master
