<<<<<<< HEAD
'''
Copyright 2020 The Microsoft DeepSpeed Team.
Licensed under the MIT license.
'''

from deepspeed.runtime.config_utils import get_scalar_param
from .offload_constants import *

OFFLOAD_PARAM_KEY_DEFAULT_DICT = {
    OFFLOAD_PARAM_DEVICE: OFFLOAD_PARAM_DEVICE_DEFAULT,
    OFFLOAD_PARAM_NVME_PATH: OFFLOAD_PARAM_NVME_PATH_DEFAULT,
    OFFLOAD_PARAM_BUFFER_COUNT: OFFLOAD_PARAM_BUFFER_COUNT_DEFAULT,
    OFFLOAD_PARAM_BUFFER_SIZE: OFFLOAD_PARAM_BUFFER_SIZE_DEFAULT,
    OFFLOAD_PARAM_MAX_IN_CPU: OFFLOAD_PARAM_MAX_IN_CPU_DEFAULT,
    OFFLOAD_PARAM_PIN_MEMORY: OFFLOAD_PARAM_PIN_MEMORY_DEFAULT
}

OFFLOAD_OPTIMIZER_KEY_DEFAULT_DICT = {
    OFFLOAD_OPTIMIZER_DEVICE: OFFLOAD_OPTIMIZER_DEVICE_DEFAULT,
    OFFLOAD_OPTIMIZER_NVME_PATH: OFFLOAD_OPTIMIZER_NVME_PATH_DEFAULT,
    OFFLOAD_OPTIMIZER_BUFFER_COUNT: OFFLOAD_OPTIMIZER_BUFFER_COUNT_DEFAULT,
    OFFLOAD_OPTIMIZER_PIN_MEMORY: OFFLOAD_OPTIMIZER_PIN_MEMORY_DEFAULT,
    OFFLOAD_OPTIMIZER_PIPELINE_READ: OFFLOAD_OPTIMIZER_PIPELINE_READ_DEFAULT,
    OFFLOAD_OPTIMIZER_PIPELINE_WRITE: OFFLOAD_OPTIMIZER_PIPELINE_WRITE_DEFAULT,
    OFFLOAD_OPTIMIZER_FAST_INIT: OFFLOAD_OPTIMIZER_FAST_INIT_DEFAULT
}


def _get_offload_config(param_dict, key_default_dict):
    offload_config = {}
    for key, default_value in key_default_dict.items():
        offload_config[key] = get_scalar_param(param_dict, key, default_value)

    return offload_config


def get_offload_param_config(param_dict):
    if OFFLOAD_PARAM in param_dict and param_dict[OFFLOAD_PARAM] is not None:
        return _get_offload_config(param_dict=param_dict[OFFLOAD_PARAM],
                                   key_default_dict=OFFLOAD_PARAM_KEY_DEFAULT_DICT)

    return None


def get_default_offload_param_config():
    return OFFLOAD_PARAM_KEY_DEFAULT_DICT


def get_offload_optimizer_config(param_dict):
    if OFFLOAD_OPTIMIZER in param_dict and param_dict[OFFLOAD_OPTIMIZER] is not None:
        offload_config = _get_offload_config(
            param_dict=param_dict[OFFLOAD_OPTIMIZER],
            key_default_dict=OFFLOAD_OPTIMIZER_KEY_DEFAULT_DICT)
        offload_config[OFFLOAD_OPTIMIZER_PIPELINE] = offload_config[
            OFFLOAD_OPTIMIZER_PIPELINE_READ] or offload_config[
                OFFLOAD_OPTIMIZER_PIPELINE_WRITE]
        return offload_config

    return None


def get_default_offload_optimizer_config():
    return OFFLOAD_OPTIMIZER_KEY_DEFAULT_DICT
=======
'''Copyright The Microsoft DeepSpeed Team'''
"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from pydantic import Field, validator
from enum import Enum
from pathlib import Path
from deepspeed.runtime.config_utils import DeepSpeedConfigModel, pp_int


class OffloadDeviceEnum(str, Enum):
    """ Enum for valid offload devices """
    none = "none"
    cpu = "cpu"
    nvme = "nvme"


class DeepSpeedZeroOffloadParamConfig(DeepSpeedConfigModel):
    """ Set options for parameter offload. Valid only with stage 3. """

    device: OffloadDeviceEnum = "none"
    """
    Device memory to offload model parameters. Supported options are `cpu` and
    `nvme`.
    """

    nvme_path: Path = None
    """ Filesystem path for NVMe device for parameter offloading. """

    buffer_count: int = Field(5, ge=0)
    """ Number of buffers in buffer pool for parameter offloading to NVMe. """

    buffer_size: int = Field(pp_int(1e8), ge=0)
    """ Size of buffers in buffer pool for parameter offloading to NVMe. """

    max_in_cpu: int = Field(pp_int(1e9), ge=0)
    """
    Number of parameter elements to maintain in CPU memory when offloading to
    NVMe is enabled.
    """

    pin_memory: bool = False
    """
    Offload to page-locked CPU memory. This could boost throughput at the cost
    of extra memory overhead.
    """


class DeepSpeedZeroOffloadOptimizerConfig(DeepSpeedConfigModel):
    """ Set options for optimizer offload. Valid with stage 1, 2, and 3. """

    device: OffloadDeviceEnum = "none"
    """
    Device memory to offload optimizer state. Supported options are `cpu` and
    `nvme`. Optimizer computation is offload to CPU regardless of device option.
    """

    nvme_path: Path = None
    """ Filesystem path for NVMe device for optimizer state offloading. """

    buffer_count: int = Field(4, ge=0)
    """
    Number of buffers in buffer pool for optimizer state offloading to NVMe.
    This should be at least the number of states maintained per parameter by
    the optimizer. For example, Adam optimizer has 4 states (parameter,
    gradient, momentum, and variance).
    """

    pin_memory: bool = False
    """
    Offload to page-locked CPU memory. This could boost throughput at the cost
    of extra memory overhead.
    """

    pipeline_read: bool = False
    """
    For tile-based optimizer step processing, overlap read of next tile with
    computation of current tile. Used in ZeRO-Infinity.
    """

    pipeline_write: bool = False
    """
    For tile-based optimizer step processing, overlap write of previous tile
    with computation of current tile.
    """

    fast_init: bool = False
    """ Enable fast optimizer initialization when offloading to NVMe. """
    @validator("pipeline_read", "pipeline_write", always=True)
    def set_pipeline(cls, field_value, values):
        values["pipeline"] = field_value or values.get("pipeline", False)
        return field_value
>>>>>>> master
