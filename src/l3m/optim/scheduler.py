# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""Adapted from:
- https://github.com/facebookresearch/omnivore/blob/main/omnivision/optim/omni_optimizer.py
"""

import math
from collections.abc import Callable

from fvcore.common.param_scheduler import ParamScheduler
from torch import optim

__all__ = ["step_scheduler"]


def step_scheduler(
    lr_scheduler: optim.lr_scheduler.LRScheduler | Callable[[float], float],
    optimizer: optim.Optimizer,
    where: float,
) -> None:
    """Steps the learning rate scheduler.

    Args:
        lr_scheduler: scheduler to step.
            Only used when the scheduler is **not** a :class:`~torch.optim.lr_scheduler.LRScheduler`:
        optimizer: optimizer defining the parameter groups.
        where: value in :math:`[0, 1)` corresponding to the percentage of training completed.
    """

    if isinstance(lr_scheduler, optim.lr_scheduler.LRScheduler):
        lr_scheduler.step()
    else:
        for param_group in optimizer.param_groups:
            new_value = lr_scheduler(where)
            if "lr_scale" in param_group:
                param_group["lr"] = new_value * param_group["lr_scale"]
            else:
                param_group["lr"] = new_value


class InvPowerScheduler(ParamScheduler):
    """InversePower scheduler

    Gives a learning rate with equation lr(t) = start_value * (offset + t) ** power / offset ** power
    where offset is computed such that lr(t=1) = end_value.
    This way, lr(t=0) = start_value and lr(t=1) = end_value.

    Args:
        start_value: initial learning rate value
        end_value: final learning rate value
        power: power in the equation to compute the current learning rate. Must be negative.
    """

    def __init__(
        self,
        start_value: float,
        end_value: float,
        power: float = -0.5,
    ):
        assert end_value < start_value, "end_value must be smaller than start_value"
        assert power < 0, "power must be negative"
        self._start_value = start_value
        self._end_value = end_value
        self._power = power
        # Compute offset so that lr(t=1) = end_value
        self._offset = 1.0 / ((end_value / start_value) ** (1.0 / power) - 1.0)

    def __call__(self, where: float) -> float:
        return self._start_value * (self._offset + where) ** self._power / self._offset**self._power


class HalfCosineParamScheduler(ParamScheduler):
    """Cosine decay over half period

    Args:
        start_value: initial learning rate value
        end_value: final learning rate value
    """

    def __init__(
        self,
        start_value: float,
        end_value: float,
    ):
        self._start_value = start_value
        self._end_value = end_value

    def __call__(self, where: float) -> float:
        return self._end_value + 0.5 * (self._start_value - self._end_value) * (1 + math.cos(math.pi * where / 2.0))


class ConstantParamScheduler(ParamScheduler):
    """Returns a constant value for a param.

    Args:
        value: constant learning rate value.
    """

    def __init__(self, value: float):
        self._value = value

    def __call__(self, where: float) -> float:
        assert where <= 1.0, f"`where` in ParamScheduler must be in [0, 1]: got {where}"
        return self._value
