# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""Parameter efficient fine-tuning modules."""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from l3m.model.layers.attention import EfficientAttention

__all__ = ["Adapter", "LoraAttention"]


class Adapter(nn.Module):
    """Adds an Adapter layer for PEFT (Parameter-Efficient Fine-Tuning).

    This module implements a simple adapter layer consisting of two linear layers
    with an activation function in between. It's designed to be inserted into
    existing models to enable efficient fine-tuning with a minimal number of
    trainable parameters.

    Args:
        embed_dim: The input and output embedding dimension.
        adapter_hidden_dim: The hidden dimension of the adapter layer.
        act_layer: The activation function to use in the adapter layer.
    """

    def __init__(
        self,
        embed_dim: int,
        adapter_hidden_dim: int,
        act_layer: Callable[[], nn.Module] = nn.GELU,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=adapter_hidden_dim),
            act_layer(),
            nn.Linear(in_features=adapter_hidden_dim, out_features=embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x


# TODO: use the new attention layer? then we would need three linear layers instead of the merged one.
class LoraAttention(EfficientAttention):
    """Attention module with LoRA (Low-Rank Adaptation).

    Args:
        lora_rank: Rank of the LoRA matrices.
        qkv_flags: Flags to enable LoRA for Q, K, and V matrices respectively.
        dim: Dimension of the input.
        qkv_bias: Whether to use bias for the Q, K, and V matrices.
        use_bias: Whether to use bias for the output projection matrix.
        kwargs: EfficientAttention kwargs.
    """

    def __init__(
        self,
        lora_rank: int = 8,
        qkv_flags: tuple[bool, bool, bool] = (True, True, True),
        **kwargs: Any,
    ):
        import loralib

        super().__init__(**kwargs)
        dim = kwargs["dim"]
        self.qkv = loralib.MergedLinear(
            dim,
            dim * 3,
            bias=kwargs["qkv_bias"],
            r=lora_rank,
            enable_lora=list(qkv_flags),
        )

        if all(qkv_flags):  # Turn on if all Q, K, and V are True
            self.proj = loralib.Linear(dim, dim, bias=kwargs["use_bias"], r=lora_rank)
