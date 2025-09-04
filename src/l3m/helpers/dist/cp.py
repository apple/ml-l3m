# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""Context Parallel (CP) helpers.

.. warning::

    This module is experimental and has not been as thoroughly tested as other modules.
"""

import contextlib
import fnmatch
from collections.abc import Generator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Literal

import torch
from hydra.utils import instantiate
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from l3m.constants.typing import DATA_DICT

__all__ = [
    "CPInputPlan",
    "CPParameterPlan",
    "CPPlan",
    "build_cp_plan",
    "create_context_parallel_ctx",
    "get_train_context",
]

# 'allgather' means to all-gather all kv shards on ranks after the first sub-SDPA computation,
# 'alltoall' means to all-to-all shuffle the kv shards.
RotationMethod = Literal["allgather", "alltoall"]


@dataclass
class CPInputPlan:
    """Context Parallelism input plan.

    Args:
        read_key: The data_dict item (torch.Tensor) to shard along the seq dim in context parallel.
        seq_dim: The dimension of the tensor buffer to shard along.
        no_restore: Whether the tensor buffer should not be restored after the cp context exits.
    """

    read_key: str
    seq_dim: int
    no_restore: bool

    def get_cp_buffer(self, data_dict: DATA_DICT) -> torch.Tensor:
        input_tensor = data_dict[self.read_key]
        return input_tensor

    def get_cp_seq_dim(self) -> int:
        return self.seq_dim

    def get_cp_no_restore_buffer(self, data_dict: DATA_DICT) -> torch.Tensor | None:
        if self.no_restore:
            return self.get_cp_buffer(data_dict=data_dict)
        return None


@dataclass
class CPParameterPlan:
    """Context Parallelism parameter plan.

    Args:
        module_pattern: A string matching pattern for selecting the module buffers to shard along
            the seq dim in context parallel.
        seq_dim: The dimension of the module buffer to shard along.
        no_restore: Whether the module buffer should not be restored after the cp context exits.
    """

    module_pattern: str
    seq_dim: int
    no_restore: bool

    def get_cp_buffers(self, model_parts: list[Any]) -> list[torch.Tensor]:
        module_buffers = []
        for model_part in model_parts:
            for name, param in model_part.named_buffers():
                if fnmatch.fnmatch(name, self.module_pattern):
                    module_buffers.append(param)

        return module_buffers

    def get_cp_seq_dim(self, model_parts: list[Any]) -> list[int]:
        return [self.seq_dim] * len(model_parts)

    def get_cp_no_restore_buffer(self, model_parts: list[Any]) -> set[torch.Tensor]:
        if self.no_restore:
            return set(self.get_cp_buffers(model_parts=model_parts))
        return set()


@dataclass
class CPPlan:
    """Context Parallelism plan.

    Args:
        cp_input_plans: List of context parallel input plans.
        cp_parameter_plans: List of context parallel parameter plans.
        cp_rotate_method: The collection to use in context parallel SPDA for kv shards exchaning.
    """

    cp_input_plans: list[CPInputPlan]
    cp_parameter_plans: list[CPParameterPlan]
    cp_rotate_method: RotationMethod

    def get_cp_buffers(
        self,
        data_dict: DATA_DICT,
        model_parts: list[Any],
    ) -> list[torch.Tensor]:
        input_buffers = []
        for cp_input_plan in self.cp_input_plans:
            input_buffer = cp_input_plan.get_cp_buffer(data_dict=data_dict)
            input_buffers.append(input_buffer)

        parameter_buffers = []
        for cp_parameter_plan in self.cp_parameter_plans:
            model_parts_parameter_buffer = cp_parameter_plan.get_cp_buffers(
                model_parts=model_parts,
            )
            parameter_buffers.extend(model_parts_parameter_buffer)

        cp_buffers = input_buffers + parameter_buffers
        return cp_buffers

    def get_seq_dims(self, model_parts: list[Any]) -> list[int]:
        input_seq_dims = []
        for cp_input_plan in self.cp_input_plans:
            input_seq_dim = cp_input_plan.get_cp_seq_dim()
            input_seq_dims.append(input_seq_dim)

        parameter_seq_dims = []
        for cp_parameter_plan in self.cp_parameter_plans:
            model_parts_parameter_seq_dims = cp_parameter_plan.get_cp_seq_dim(
                model_parts=model_parts,
            )
            parameter_seq_dims.extend(model_parts_parameter_seq_dims)

        cp_seq_dims = input_seq_dims + parameter_seq_dims
        return cp_seq_dims

    def get_cp_no_restore_buffers(
        self,
        data_dict: DATA_DICT,
        model_parts: list[Any],
    ) -> set[torch.Tensor]:
        input_no_restore_buffers = set()
        for cp_input_plan in self.cp_input_plans:
            input_no_restore_buffer = cp_input_plan.get_cp_no_restore_buffer(data_dict=data_dict)
            if input_no_restore_buffer is not None:
                input_no_restore_buffers.add(input_no_restore_buffer)

        parameter_no_restore_buffers = set()
        for cp_parameter_plan in self.cp_parameter_plans:
            model_parts_parameter_no_restore_buffers = cp_parameter_plan.get_cp_no_restore_buffer(
                model_parts=model_parts
            )
            parameter_no_restore_buffers.update(model_parts_parameter_no_restore_buffers)

        cp_seq_dims = input_no_restore_buffers.union(parameter_no_restore_buffers)
        return cp_seq_dims


def build_cp_plan(exp_cfg: dict[Any, Any]) -> CPPlan | None:
    if "cp_plan" not in exp_cfg:
        return None

    cp_plan_cfg = exp_cfg["cp_plan"]
    cp_plan = CPPlan(
        cp_input_plans=[instantiate(cfg) for cfg in cp_plan_cfg["cp_input_plans"]],
        cp_parameter_plans=[instantiate(cfg) for cfg in cp_plan_cfg["cp_parameter_plans"]],
        cp_rotate_method=cp_plan_cfg["cp_rotate_method"],
    )
    return cp_plan


def create_context_parallel_ctx(
    cp_mesh: DeviceMesh,
    cp_plan: CPPlan,
    data_dict: DATA_DICT,
    model_parts: list[nn.Module],
):
    """Creates a context parallel context for context parallel execution.

    Args:
        cp_mesh: The device mesh for context parallelism.
        cp_plan: The context parallel plan.
        data_dict: The input data dictionary.
        model_parts: A list of model modules to parallelize.

    Returns:
        The context parallel context.
    """

    try:
        from torch.distributed.tensor.experimental import context_parallel
        from torch.distributed.tensor.experimental._attention import set_rotate_method
    except ImportError:
        raise RuntimeError(
            f"PyTorch version {torch.__version__} does not include the experimental "
            "Context Parallel API. Please update to a newer version."
        ) from None

    cp_buffers = cp_plan.get_cp_buffers(
        data_dict=data_dict,
        model_parts=model_parts,
    )
    cp_seq_dims = cp_plan.get_seq_dims(
        model_parts=model_parts,
    )
    cp_no_restore_buffers = cp_plan.get_cp_no_restore_buffers(
        data_dict=data_dict,
        model_parts=model_parts,
    )
    cp_rotate_method = cp_plan.cp_rotate_method

    set_rotate_method(cp_rotate_method)
    return context_parallel(
        cp_mesh,
        buffers=cp_buffers,
        buffer_seq_dims=cp_seq_dims,
        no_restore_buffers=cp_no_restore_buffers,
    )


def get_train_context(enable_loss_parallel: bool) -> AbstractContextManager:
    """Returns a context manager for training with optional context parallelism and loss parallelism.

    Args:
        enable_loss_parallel: Whether to enable loss parallelism.

    Returns:
        A context manager for training.
    """

    @contextlib.contextmanager
    def context(cp_context: Generator[None, None, None] | None = None):
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())

            if cp_context is not None:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                stack.enter_context(sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]))
                stack.enter_context(cp_context)

            yield

    return context
