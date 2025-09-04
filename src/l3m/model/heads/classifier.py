# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from timm.layers.weight_init import variance_scaling_

from l3m.constants.typing import DATA_DICT
from l3m.model.layers.normalization import LAYER_NORM, LayerNormFP32
from l3m.model.meta_models import ReadWriteBlock
from l3m.model.trunks.transformer import Block

__all__ = [
    "LinearClassifier",
    "TransformerClassifier",
    "AttentionPoolingClassifier",
    "SimplifiedAttentionPoolingClassifier",
]

_LAYER_NORM_FP32 = partial(LayerNormFP32, eps=1e-5)


class LinearClassifier(ReadWriteBlock):
    """A linear classifier head with optional layer normalization and batch normalization.

    Applies a linear transformation to the input features, optionally followed by
    batch normalization and/or layer normalization. Supports different weight
    initialization schemes.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        use_layernorm: Whether to use layer normalization.
        use_batchnorm: Whether to use batch normalization.
        weight_init_style: Weight initialization style.
        encoder_num_blocks: Number of encoder blocks (required for "fan_in_depth_scaled" init).
        init_std: Standard deviation for truncated normal initialization.
        use_bias: Whether to use bias in the linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_layernorm: bool = False,
        use_batchnorm: bool = False,
        weight_init_style: Literal["xavier", "trunc_normal", "fan_in_depth_scaled", "zero", "normal"] = "xavier",
        encoder_num_blocks: int = None,
        init_std: float = 0.02,
        use_bias: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.norm = _LAYER_NORM_FP32(in_features) if use_layernorm else nn.Identity()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=use_bias)
        self.bn = (
            torch.nn.BatchNorm1d(kwargs["in_features"], affine=False, eps=1e-6) if use_batchnorm else nn.Identity()
        )

        self.weight_init_style = weight_init_style
        self.encoder_num_blocks = encoder_num_blocks

        if self.weight_init_style == "fan_in_depth_scaled":
            assert encoder_num_blocks is not None, (
                "`encoder_num_blocks` is required when using `fan_in_depth_scaled` init for the linear head."
            )

        self.init_std = init_std

        self.init_weights()

    def init_weights(self) -> None:
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            if self.weight_init_style == "xavier":
                torch.nn.init.xavier_uniform_(m.weight)
            elif self.weight_init_style == "trunc_normal":
                trunc_normal_(m.weight, std=self.init_std)
            elif self.weight_init_style == "fan_in_depth_scaled":
                a = 1 / (1 + self.encoder_num_blocks)
                variance_scaling_(
                    m.weight,
                    scale=a,
                    mode="fan_in",
                    distribution="truncated_normal",
                )
            elif self.weight_init_style == "zero":
                # iGPT reported zero initializing last logit projection layers
                torch.nn.init.constant(m.weight, 0.0)
            elif self.weight_init_style == "normal":
                torch.nn.init.normal_(m.weight, mean=0.0, std=self.init_std)
            else:
                raise ValueError(f"Undefined initialization {self.weight_init_style}.")

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, data_dict: DATA_DICT, **_: Any) -> DATA_DICT:
        data_dict[self.write_key] = self.linear(self.bn(self.norm(data_dict[self.read_key])))
        return data_dict


class TransformerClassifier(ReadWriteBlock):
    """Transformer-based classifier head.

    Applies a transformer block, normalizes, and then applies a linear layer for classification.

    Args:
        in_features: Number of input features.
        out_features: Number of output features (classes).
        attn_target: Attention target function.
        mlp_ratio: Ratio of mlp hidden dimension to input dimension.
        use_layernorm: Whether to use LayerNorm.
        use_batchnorm: Whether to use BatchNorm.
        norm_layer: Normalization layer.
        weight_init_style: Weight initialization style.
        init_std: Standard deviation for truncated normal initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        attn_target: Callable,
        mlp_ratio: int = 4,
        use_layernorm: bool = False,
        use_batchnorm: bool = False,
        norm_layer: Callable[[int], nn.Module] = LAYER_NORM,
        weight_init_style: Literal["xavier", "trunc_normal"] = "xavier",
        init_std: float = 0.02,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.block = Block(
            attn_target=attn_target,
            embed_dim=in_features,
            mlp_hidden_dim=int(mlp_ratio * in_features),
            norm_layer=norm_layer,
        )

        self.norm = nn.LayerNorm(in_features) if use_layernorm else nn.Identity()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.bn = (
            torch.nn.BatchNorm1d(kwargs["in_features"], affine=False, eps=1e-6) if use_batchnorm else nn.Identity()
        )

        self.weight_init_style = weight_init_style
        self.init_std = init_std

        self.init_weights()

    def init_weights(self) -> None:
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

    def forward(self, data_dict: DATA_DICT, **_: Any) -> DATA_DICT:
        x = self.block(data_dict[self.read_key]).mean(dim=1)
        data_dict[self.write_key] = self.linear(self.bn(self.norm(x)))
        return data_dict


class AttentionPoolingClassifier(ReadWriteBlock):
    """A classifier that applies attention pooling to an input sequence to generate a class representation.
    This representation is then used for classification via a linear layer.

    Adapted from: https://github.com/facebookresearch/deit/blob/main/cait_models.py

    Args:
        dim: Input dimension.
        out_features: Number of output features (classes).
        num_heads: Number of attention heads (default: 8).
        qkv_bias: Whether to use bias in QKV linear layers (default: False).
        qk_scale: Scale factor for query-key dot product (default: None).
        attn_drop: Dropout probability for attention weights (default: 0.0).
        proj_drop: Dropout probability for projection layer (default: 0.0).
        use_layernorm: Whether to use LayerNorm (default: False).
        use_batchnorm: Whether to use BatchNorm (default: False).
        sync_batchnorm: Whether to use SyncBatchNorm (default: False).
        create_queries: Whether to create new query tokens, or use existing (default: True).
        weight_init_style: Weight initialization style (default: "xavier").
        init_std: Standard deviation for weight initialization (default: 0.02).
        num_queries: Number of query tokens (default: 1).
    """

    def __init__(
        self,
        dim: int,
        out_features: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_layernorm: bool = False,
        use_batchnorm: bool = False,
        sync_batchnorm: bool = False,
        create_queries: bool = True,
        weight_init_style: Literal["xavier", "trunc_normal"] = "xavier",
        init_std: float = 0.02,
        num_queries: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.cls_token = nn.Parameter(torch.empty(1, num_queries, dim)) if create_queries else None
        self.linear = nn.Linear(in_features=dim, out_features=out_features)
        self.norm = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        batchnorm_class = torch.nn.SyncBatchNorm if sync_batchnorm else torch.nn.BatchNorm1d
        self.bn = batchnorm_class(dim, affine=False, eps=1e-6) if use_batchnorm else nn.Identity()

        self.num_queries = num_queries
        self.weight_init_style = weight_init_style
        self.init_std = init_std

        self.init_weights()

    def init_weights(self) -> None:
        if self.cls_token is not None:
            torch.nn.init.normal_(self.cls_token, std=0.02)

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

    def forward(self, data_dict: DATA_DICT, **_: Any) -> DATA_DICT:
        x = data_dict[self.read_key]
        B, N, C = x.shape

        x = self.bn(x.transpose(-2, -1)).transpose(-2, -1)
        if self.cls_token is None:
            cls_token = x[:, 0]  # sequence already has the class token
            x = x[:, 1:, :]
            N = N - 1
        else:
            cls_token = self.cls_token.expand(B, -1, -1)  # newly created class token

        q = self.q(cls_token).reshape(B, self.num_queries, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, self.num_queries, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        x_cls = x_cls.mean(dim=1)

        data_dict[self.write_key] = self.linear(self.norm(x_cls))
        return data_dict


class SimplifiedAttentionPoolingClassifier(ReadWriteBlock):
    """Simplified version of the AttentionPollingClassifier.

    Adapted from: https://github.com/facebookresearch/deit/blob/main/cait_models.py

    Args:
        dim: Input dimension.
        out_features: Number of output features.
        num_heads: Number of attention heads.
        qkv_bias: Whether to use bias in QKV linear layers.
        qk_scale: Scale factor for query-key dot product.
        use_layernorm: Whether to use LayerNorm.
        use_batchnorm: Whether to use BatchNorm.
        sync_batchnorm: Whether to use SyncBatchNorm.
        create_queries: Whether to create new query tokens, or use existing.
        weight_init_style: Weight initialization style.
        init_std: Standard deviation for weight initialization.
        num_queries: Number of query tokens.
    """

    def __init__(
        self,
        dim: int,
        out_features: int,
        num_heads: int = 12,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        use_layernorm: bool = False,
        use_batchnorm: bool = False,
        sync_batchnorm: bool = False,
        create_queries: bool = True,
        weight_init_style: Literal["xavier", "trunc_normal"] = "xavier",
        init_std: float = 0.02,
        num_queries: int = 1,
        **kwargs: Any,
    ):
        if num_queries != 1:
            assert create_queries, "only supports a single query (the cls token) if not creating new ones"

        super().__init__(**kwargs)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.cls_token = nn.Parameter(torch.empty(1, num_queries, dim)) if create_queries else None
        self.linear = nn.Linear(in_features=dim, out_features=out_features)
        self.norm = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        batchnorm_class = torch.nn.SyncBatchNorm if sync_batchnorm else torch.nn.BatchNorm1d
        self.bn = batchnorm_class(dim, affine=False, eps=1e-6) if use_batchnorm else nn.Identity()

        self.num_queries = num_queries
        self.weight_init_style = weight_init_style
        self.init_std = init_std

        self.init_weights()

    def init_weights(self) -> None:
        if self.cls_token is not None:
            torch.nn.init.normal_(self.cls_token, std=0.02)

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

    def forward(self, data_dict: DATA_DICT, **_: Any) -> DATA_DICT:
        x = data_dict[self.read_key]
        B, N, C = x.shape

        x = self.bn(x.transpose(-2, -1)).transpose(-2, -1)
        if self.cls_token is None:
            cls_token = x[:, 0]  # already present class token
            x = x[:, 1:, :]
            N = N - 1
        else:
            cls_token = self.cls_token.expand(B, -1, -1)  # newly created class token

        q = cls_token.reshape(B, self.num_queries, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, self.num_queries, C)
        x_cls = x_cls.mean(dim=1)

        data_dict[self.write_key] = self.linear(self.norm(x_cls))
        return data_dict
