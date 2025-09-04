# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Any

import torch
import torch.nn as nn

from l3m.helpers.moe import utils as moe_utils
from l3m.model.meta_models import ReadWriteBlock

__all__ = ["MoEBalancingLoss"]


class MoEBalancingLoss(ReadWriteBlock):
    """A class to compute the load balancing loss for a Mixture-of-Experts (MoE) model."""

    def forward(self, *_: Any, model: nn.Module, **__: Any) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            _: Positional arguments passed to the forward method. Unused.
            model: The MoE model for which the load balancing loss is computed.
            __: Keyword arguments passed to the forward method. Unused.

        Returns:
            A tuple consisting of:

            - A tensor representing the load balancing loss.
            - A dictionary with the load balancing loss as a scalar value under the key "moe/load_balancing_loss".
        """
        load_balancing_loss = moe_utils.get_load_balancing_loss_given_moe(model)
        return load_balancing_loss, {"moe/load_balancing_loss": load_balancing_loss.item()}
