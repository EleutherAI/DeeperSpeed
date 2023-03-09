"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
<<<<<<< HEAD
from .cpu_adam import CPUAdamBuilder
from .fused_adam import FusedAdamBuilder
from .fused_lamb import FusedLambBuilder
from .sparse_attn import SparseAttnBuilder
from .transformer import TransformerBuilder
from .stochastic_transformer import StochasticTransformerBuilder
from .utils import UtilsBuilder
from .async_io import AsyncIOBuilder
from .builder import get_default_compute_capatabilities

# TODO: infer this list instead of hard coded
# List of all available ops
__op_builders__ = [
    CPUAdamBuilder(),
    FusedAdamBuilder(),
    FusedLambBuilder(),
    SparseAttnBuilder(),
    TransformerBuilder(),
    StochasticTransformerBuilder(),
    UtilsBuilder(),
    AsyncIOBuilder()
]
ALL_OPS = {op.name: op for op in __op_builders__}
=======
import sys
import os
import pkgutil
import importlib

from .builder import get_default_compute_capabilities, OpBuilder

# Do not remove, required for abstract accelerator to detect if we have a deepspeed or 3p op_builder
__deepspeed__ = True

# List of all available op builders from deepspeed op_builder
try:
    import deepspeed.ops.op_builder  # noqa: F401
    op_builder_dir = "deepspeed.ops.op_builder"
except ImportError:
    op_builder_dir = "op_builder"

__op_builders__ = []

this_module = sys.modules[__name__]


def builder_closure(member_name):
    if op_builder_dir == "op_builder":
        # during installation time cannot get builder due to torch not installed,
        # return closure instead
        def _builder():
            from deepspeed.accelerator import get_accelerator
            builder = get_accelerator().create_op_builder(member_name)
            return builder

        return _builder
    else:
        # during runtime, return op builder class directly
        from deepspeed.accelerator import get_accelerator
        builder = get_accelerator().get_op_builder(member_name)
        return builder


# reflect builder names and add builder closure, such as 'TransformerBuilder()' creates op builder wrt current accelerator
for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(this_module.__file__)]):
    if module_name != 'all_ops' and module_name != 'builder':
        module = importlib.import_module(f".{module_name}", package=op_builder_dir)
        for member_name in module.__dir__():
            if member_name.endswith(
                    'Builder'
            ) and member_name != "OpBuilder" and member_name != "CUDAOpBuilder":
                # assign builder name to variable with same name
                # the following is equivalent to i.e. TransformerBuilder = "TransformerBuilder"
                this_module.__dict__[member_name] = builder_closure(member_name)
>>>>>>> master
