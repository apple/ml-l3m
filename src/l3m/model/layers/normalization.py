# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import numbers
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LAYER_NORM", "LayerNormFP32", "RMSNorm", "LPLayerNorm"]

LAYER_NORM = partial(nn.LayerNorm, eps=1e-6)


class LayerNormFP32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=torch.float32):
            input_dtype = x.dtype
            x = x.to(torch.float32)
            return super().forward(x).to(input_dtype)


class RMSNorm(torch.nn.Module):
    """RMS normalization layer.

    Adapted from: https://github.com/facebookresearch/llama/blob/main/llama/model.py

    Args:
        dim: The dimension of the input tensor.
        eps: A small value added to the denominator for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight.data.fill_(1.0)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xdtype = x.dtype
        output = self._norm(x.float()).type_as(x)
        return (output * self.weight).to(xdtype)


class LPLayerNorm(nn.Module):
    """Low Precision Layer Norm adapted to support the independent removal of gain and bias.

    Adapted from:
    https://github.com/mosaicml/composer/blob/6acca4c70425455be7280a5459dbf02e1ac5591d/composer/algorithms/low_precision_layernorm/low_precision_layernorm.py#L63

    Args:
        normalized_shape: input shape from an expected input of size.
        eps: a value added to the denominator for numerical stability.
        elementwise_gain: whether to have element-wise gain.
        elementwise_bias: whether to have element-wise bias.
    """

    def __init__(
        self,
        normalized_shape: int | list[int] | torch.Size,
        eps: float = 0.00001,
        elementwise_gain: bool = True,
        elementwise_bias: bool = True,
    ):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_gain = elementwise_gain
        self.elementwise_bias = elementwise_bias

        if self.elementwise_gain:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter("weight", None)

        if self.elementwise_bias:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.elementwise_gain:
            self.weight.fill_(1.0)
        if self.elementwise_bias:
            self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=torch.float32):
            input_dtype = x.dtype
            x = x.to(torch.float32)
            return F.layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            ).to(input_dtype)

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_gain={elementwise_gain}, "
            "elementwise_bias={elementwise_bias}".format(**self.__dict__)
        )
