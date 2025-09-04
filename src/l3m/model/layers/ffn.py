# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

__all__ = ["MLP", "GatedFFN", "SwiGLUFFN", "GeGELU"]


class MLP(nn.Module):
    """A Multi-Layer Perceptron (MLP) block.

    Args:
        in_features: Number of input features.
        hidden_features: Number of hidden features.
        out_features: Number of output features.
        act_layer: Activation function.
        use_bias: Whether to use bias in linear layers.
        drop: Dropout probability.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        use_bias: bool = True,
        drop: float = 0.0,
        norm_layer: Callable[[int], nn.Module] | None = None,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=use_bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=use_bias)
        self.drop = nn.Dropout(drop)
        if norm_layer is not None:
            self.norm = norm_layer(out_features)
        else:
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.norm(x)
        return x


class GatedFFN(nn.Module):
    """A gated feedforward network (FFN) layer.

    Args:
        in_features: Number of input features.
        hidden_features: Number of hidden features.
        act_layer: Activation function.
        multiple_of: Make hidden layer size divisible by this value.
        use_bias: Whether to use bias in linear layers.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_layer: Callable[[], nn.Module],
        multiple_of: int = 256,
        use_bias: bool = True,
        **_: Any,
    ):
        super().__init__()

        hidden_features = int(2 * hidden_features / 3)
        hidden_features = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=use_bias)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=use_bias)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=use_bias)

        self.act_layer = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_layer(self.fc1(x)) * self.fc3(x))


class SwiGLUFFN(GatedFFN):
    """Creates a SwiGLU FFN by setting F.silu as the default activation function."""

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.pop("act_layer", None)
        super().__init__(*args, act_layer=nn.SiLU, **kwargs)


class GeGELU(GatedFFN):
    """Creates a GeGELU FFN by setting F.gelu as the default activation function."""

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.pop("act_layer", None)
        super().__init__(*args, act_layer=nn.GELU, **kwargs)
