# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""Adapted from:

- https://github.com/facebookresearch/moco-v3
- https://github.com/facebookresearch/mae/blob/main/util/lars.py
"""

from collections.abc import Iterable

import torch
from torch import optim

__all__ = ["LARS"]


class LARS(optim.Optimizer):
    """LARS optimizer.

    No rate scaling or weight decay is applied for parameters <= 1D.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr: learning rate.
        weight_decay: weight decay (L2 penalty).
        momentum: momentum factor.
        trust_coefficient: trust coefficient used to calculate the adaptive learning rate.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, torch.Tensor]],
        lr: float = 0.0,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        trust_coefficient: float = 1e-3,
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "trust_coefficient": trust_coefficient,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self) -> None:  # type: ignore[override]
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1:  # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g["weight_decay"])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0,
                            (g["trust_coefficient"] * param_norm / update_norm),
                            one,
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)
                p.add_(mu, alpha=-g["lr"])
