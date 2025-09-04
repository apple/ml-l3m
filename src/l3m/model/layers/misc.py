# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Any, Literal

import torch
import torch.nn as nn
from timm.layers.weight_init import trunc_normal_, variance_scaling_

from l3m.constants.typing import DATA_DICT
from l3m.model.meta_models import ReadWriteBlock

__all__ = [
    "GenericSequential",
    "GenericIdentity",
    "PerceiverResampler",
    "LinearWithCustomInit",
]


class GenericSequential(nn.Sequential):
    """Wrapper for nn.Sequential to accept multiple args but process only the first one."""

    def forward(self, x: torch.Tensor, **_: Any) -> torch.Tensor:
        return super().forward(x)


class GenericIdentity(nn.Identity):
    """Wrapper for nn.Identity to accept multiple args but process only the first one."""

    def forward(self, x: torch.Tensor, *_: Any, **__: Any) -> torch.Tensor:
        return super().forward(x)


class PerceiverResampler(ReadWriteBlock):
    """A Perceiver Resampler module that maps an input sequence to a fixed number of queries.

    Adapted from: https://github.com/facebookresearch/deit/blob/main/cait_models.py

    Args:
        dim: Input and output dimension.
        num_heads: Number of attention heads.
        qkv_bias: Whether to include bias in QKV linear projections.
        qk_scale: Override default `qk` scaling value.
        weight_init_style: Weight initialization style.
        init_std: Standard deviation for weight initialization (trunc_normal only).
        num_queries: Number of output queries.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = False,
        qk_scale: float = None,
        weight_init_style: Literal["xavier", "trunc_normal"] = "xavier",
        init_std: float = 0.02,
        num_queries: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.out_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.queries = nn.Parameter(torch.empty(1, num_queries, dim))
        self.num_queries = num_queries
        self.weight_init_style = weight_init_style
        self.init_std = init_std

        self.init_weights()

    def init_weights(self) -> None:
        torch.nn.init.normal_(self.queries, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            if self.weight_init_style == "xavier":
                torch.nn.init.xavier_uniform_(m.weight)
            elif self.weight_init_style == "trunc_normal":
                trunc_normal_(m.weight, std=self.init_std)
            else:
                raise ValueError(f"Undefined initialization {self.weight_init_style}.")

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        x = data_dict[self.read_key]

        B, N, C = x.shape
        # expand learnable queries so that they match the current batch size
        queries = self.queries.expand(B, -1, -1)
        q = queries.reshape(B, self.num_queries, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, self.num_queries, C)
        out = self.out_proj(out)

        data_dict[self.write_key] = out
        return data_dict


class LinearWithCustomInit(nn.Linear):
    """Linear layer with custom initialization function.

    Args:
        args: Linear layer default arguments.
        weight_init_style: Weight initialization style.
        init_std: Standard deviation for weight initialization (trunc_normal only).
        kwargs: Linear layer default kwargs.
    """

    def __init__(
        self,
        *args,
        weight_init_style: Literal["trunc_normal", "kaiming", "fan_in", "pytorch"] = "pytorch",
        init_std: float = 0.02,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.weight_init_style = weight_init_style
        self.init_std = init_std

        self.init_weights()

    def init_weights(self) -> None:
        if self.weight_init_style == "trunc_normal":
            trunc_normal_(self.weight, std=self.init_std)
        elif self.weight_init_style == "kaiming":
            nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        elif self.weight_init_style == "fan_in":
            variance_scaling_(self.weight, scale=1.0, mode="fan_in", distribution="normal")
        elif self.weight_init_style == "pytorch":
            self.reset_parameters()
        else:
            raise ValueError(f"Unknown weight init style: {self.weight_init_style}")

        if self.bias is not None:
            nn.init.zeros_(self.bias)
