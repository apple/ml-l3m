# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import logging
from typing import Any

import torch
from torch import nn, optim
from torch.amp import GradScaler

__all__ = ["NativeScaler"]

logger = logging.getLogger("l3m")


class NativeScaler:
    """Thin wrapper around the gradient scaler.

    Args:
        enabled: Whether to enable the scaler.
    """

    def __init__(
        self,
        enabled: bool = True,
    ):
        self._scaler = GradScaler("cuda", enabled=enabled)
        logger.info(f"Gradient Scaler {'Enabled' if enabled else 'Disabled'}.")

    def __call__(
        self,
        optimizer: optim.Optimizer,
        clip_grad: float | None = None,
        model: nn.Module | None = None,
        **kwargs: Any,
    ):
        """Scale the gradients.

        Args:
            optimizer: optimizer that applies the gradients.
            clip_grad: norm used to clip the gradients. If None, skip clipping.
            model: model's whose gradients to clip.
            kwargs: kwargs for step method of the wrapped gradient scaler.
        """

        if clip_grad is not None:
            assert model is not None, "No model was passed when clipping the gradients."
            # unscale the gradients of optimizer's assigned params in-place
            self._scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad, norm_type=2.0, foreach=True)

        self._scaler.step(optimizer, **kwargs)
        self._scaler.update()

    def backward(self, loss: torch.Tensor) -> None:
        self._scaler.scale(loss).backward()

    def state_dict(self) -> dict[str, Any]:
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._scaler.load_state_dict(state_dict)
