# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import torch
import torch.nn as nn

__all__ = ["QuickGELU"]


class QuickGELU(nn.Module):
    """Quick GELU implementation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)
