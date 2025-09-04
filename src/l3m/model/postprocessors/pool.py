# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from l3m.constants.typing import DATA_DICT
from l3m.model.meta_models import ReadWriteBlock

__all__ = ["GAP", "WeightedAveragePooling", "ConcatPooling"]


class GAP(ReadWriteBlock):
    """Applies global average pooling to the input tensor along a specified dimension.

    Args:
        start: Start index for slicing the input tensor. Default: 1.
        end: End index for slicing the input tensor. Default: None.
        dim: Dimension along which to perform average pooling. Default: 1.
        keepdim: Whether to keep the dimension being pooled over. Default: False.
    """

    def __init__(
        self,
        start: int = 1,
        end: int | None = None,
        dim: int = 1,
        keepdim: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.dim = dim
        self.start, self.end = start, end
        self.keepdim = keepdim

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        x = data_dict[self.read_key]
        data_dict[self.write_key] = torch.mean(x[:, self.start : self.end], dim=self.dim, keepdim=self.keepdim)
        return data_dict


class WeightedAveragePooling(ReadWriteBlock):
    """Performs a weighted average pooling operation over a sequence of tokens.

    The input is multiplied by learnable weights, and then reduced along the sequence dimension.

    Args:
        num_tokens: Number of tokens in the sequence.
        reduction: Reduction operation to apply along the sequence dimension, e.g., :func:`torch.mean`.
        non_linear_layer: Non-linear layer to apply to the weights before weighting.
    """

    def __init__(
        self,
        num_tokens: int,
        reduction: Callable[[torch.Tensor], torch.Tensor],
        non_linear_layer: Callable[[], nn.Module] = nn.Identity,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.num_tokens = num_tokens
        self.reduction = reduction
        self.non_linear_layer = non_linear_layer()

        self.weights = nn.Parameter(torch.randn(num_tokens), requires_grad=True)

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        x = data_dict[self.read_key]
        assert x.ndim == 3, f"x ({x.shape}) needs to be of shape [b, s, d]"
        assert x.size(1) == self.num_tokens, f"Missing weights (target={self.num_tokens}, current={self.num_tokens})"

        data_dict[self.write_key] = self.reduction(self.non_linear_layer(self.weights).view(1, -1, 1) * x, dim=1)
        return data_dict


class ConcatPooling(ReadWriteBlock):
    """Concatenates the embeddings of a sequence into a single vector."""

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        x = data_dict[self.read_key]
        assert x.ndim == 3, f"x ({x.shape}) needs to be of shape [b, s, d]"

        data_dict[self.write_key] = x.view(x.shape[0], -1)

        return data_dict
