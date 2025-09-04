# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from torch.distributed.device_mesh import DeviceMesh

from l3m.helpers.utils import get_module

try:
    from torch.distributed.tensor import Replicate, Shard
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        ParallelStyle,
        PrepareModuleInput,
        PrepareModuleOutput,
        RowwiseParallel,
        SequenceParallel,
        parallelize_module,
    )

    __all__ = ["TPPlan", "TPPlanFactory", "tp_parallelize", "assert_can_do_tp"]

    tp_supported = True
except ImportError:
    __all__ = ["assert_can_do_tp"]

    tp_supported = False


@dataclass
class TPPlan:
    """Tensor parallelism plan.

    Args:
        module: Target module name.
        factory: Callable that returns the TP plan for that module (use something from TPPlanFactory).
    """

    module: str
    factory: Callable[[], ParallelStyle | dict[str, ParallelStyle] | None]


class TPPlanFactory:
    @staticmethod
    def colwise_parallel_replicate():
        return ColwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
            use_local_output=False,
        )

    @staticmethod
    def colwise_parallel_shard():
        return ColwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        )

    @staticmethod
    def colwise_parallel_replicate_output():
        return ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Replicate(),
        )

    @staticmethod
    def sequence_parallel():
        return SequenceParallel()

    @staticmethod
    def patchifier_plan():
        return {
            "proj": ColwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
            "norm": SequenceParallel(),
        }

    @staticmethod
    def prepare_input_replicate_to_shard(n_dtensors: int, n_nones: int):
        return PrepareModuleInput(
            input_layouts=(
                *[Replicate() for _ in range(n_dtensors)],
                *[None] * n_nones,
            ),
            desired_input_layouts=(
                *[Shard(1) for _ in range(n_dtensors)],
                *[None] * n_nones,
            ),
            use_local_output=True,
        )

    @staticmethod
    def prepare_output_shard_to_replicate(n_dtensors: int, n_nones: int):
        return PrepareModuleOutput(
            output_layouts=(*[Shard(1) for _ in range(n_dtensors)], *[None] * n_nones),
            desired_output_layouts=(
                *[Replicate() for _ in range(n_dtensors)],
                *[None] * n_nones,
            ),
            use_local_output=True,
        )

    @staticmethod
    def transformer_layer_plan() -> dict[str, Any]:
        return {
            # attn
            "attn": PrepareModuleInput(input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),)),
            "attn.wq": ColwiseParallel(),
            "attn.wk": ColwiseParallel(),
            "attn.wv": ColwiseParallel(),
            "attn.proj": RowwiseParallel(output_layouts=Shard(1)),
            # mlp
            "mlp": PrepareModuleInput(input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),)),
            "mlp.fc1": ColwiseParallel(),
            "mlp.fc2": RowwiseParallel(output_layouts=Shard(1)),
            "mlp.fc3": ColwiseParallel(),
            # norms
            "norm_1": SequenceParallel(),
            "norm_2": SequenceParallel(),
        }


def tp_parallelize(model: nn.Module, tp_mesh: DeviceMesh, exp_cfg: DictConfig) -> None:
    """Parallelize a model using tensor parallelism (TP).

    Args:
        model: The model to parallelize.
        tp_mesh: The device mesh for tensor parallelism.
        exp_cfg: The experiment configuration.
    """

    tp_plan = exp_cfg["tp_plan"]
    if not isinstance(tp_plan, list | ListConfig):
        tp_plan = [tp_plan]

    for plan in tp_plan:
        plan: TPPlan = instantiate(plan)
        module = get_module(model, plan.module)
        if isinstance(module, Iterable):
            for submodule in module:
                parallelize_module(
                    submodule,
                    tp_mesh,
                    plan.factory(),
                )
        else:
            parallelize_module(
                module,
                tp_mesh,
                plan.factory(),
            )


def assert_can_do_tp(exp_cfg: DictConfig) -> None:
    assert tp_supported, f"torch version={torch.__version__} < 2.5"
    assert exp_cfg["tp_plan"] is not None, "TP plan is required for TP parallelism"
    assert not exp_cfg["torch_compile"], "Compile is not supported for TP parallelism"
