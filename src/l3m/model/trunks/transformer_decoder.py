# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import logging
import math
from collections.abc import Callable
from types import MethodType
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_
from timm.layers.weight_init import variance_scaling_

from l3m.constants.typing import DATA_DICT
from l3m.model.layers.ffn import MLP
from l3m.model.layers.normalization import LAYER_NORM
from l3m.model.meta_models import ReadWriteBlock

__all__ = ["DecoderBlock", "TransformerDecoder"]

logger = logging.getLogger("l3m")


class DecoderBlock(nn.Module):
    """Transformer decoder block consisting of self-attention, cross-attention, and MLP layers.

    Args:
        dim: Input dimension.
        self_attn_target: Self-attention module.
        cross_attn_target: Cross-attention module.
        mlp_hidden_dim: Hidden dimension of the MLP.
        ffn_target: Feedforward network module.
        act_layer: Activation layer.
        norm_layer: Normalization layer.
        ffn_dropout_rate: Dropout rate for the feedforward network.
        drop_path: Drop path rate.
        use_bias: Whether to use bias in linear layers.
    """

    def __init__(
        self,
        dim: int,
        self_attn_target: Callable[[bool], nn.Module],
        cross_attn_target: Callable[[bool], nn.Module],
        mlp_hidden_dim: int,
        ffn_target: Callable[..., nn.Module] = MLP,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = LAYER_NORM,
        ffn_dropout_rate: float = 0.0,
        drop_path: float = 0.0,
        use_bias: bool = True,
    ):
        super().__init__()

        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()

        self.self_attn = self_attn_target(use_bias=use_bias)
        self.cross_attn = cross_attn_target(use_bias=use_bias)
        self.mlp = ffn_target(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=ffn_dropout_rate,
            use_bias=use_bias,
        )

        self.norm_1 = norm_layer(dim)
        self.norm_2 = norm_layer(dim)
        self.norm_3 = norm_layer(dim)

    def forward(
        self,
        x: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.drop_path(self.self_attn(self.norm_1(x), mask=mask))
        x = x + self.drop_path(
            self.cross_attn(
                queries=self.norm_2(x),
                keys=encoder_outputs,
                values=encoder_outputs,
                mask=mask,
            )
        )
        x = x + self.drop_path(self.mlp(self.norm_3(x)))
        return x


class TransformerDecoder(ReadWriteBlock):
    """Transformer decoder consisting of multiple decoder blocks.

    Args:
        encoder_output_key: Key in the data dictionary for encoder outputs.
        self_attn_target: Self-attention constructor.
        cross_attn_target: Cross-attention constructor.
        embed_dim: Embedding dimension.
        num_blocks: Number of decoder blocks.
        ffn_target: Feedforward network constructor.
        act_layer: Activation layer.
        block: Decoder block constructor.
        self_attn_mask_read_key: Key in the data dictionary for self-attention mask.
        drop_path_rate: Drop path rate.
        drop_path_type: Drop path type ("progressive" or "uniform").
        norm_layer: Normalization layer.
        mlp_ratio: Ratio of MLP hidden dimension to embedding dimension.
        mlp_hidden_dim: Hidden dimension of the MLP.
        ffn_dropout_rate: Dropout rate for the feedforward network.
        weight_init_style: Weight initialization style.
        gpt_proj_rescale: Whether to rescale projections per GPT-2.
        use_bias: Whether to use bias in linear layers.
        post_trunk_norm: Whether to apply normalization after the trunk.
    """

    def __init__(
        self,
        encoder_output_key: str,
        self_attn_target: Callable[[bool], nn.Module],
        cross_attn_target: Callable[[bool], nn.Module],
        embed_dim: int,
        num_blocks: int,
        ffn_target: Callable[..., nn.Module] = MLP,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        block: Callable[..., nn.Module] = DecoderBlock,
        self_attn_mask_read_key: str | None = None,
        drop_path_rate: float = 0.0,
        drop_path_type: Literal["progressive", "uniform"] = "progressive",
        norm_layer: Callable[[int], nn.Module] = LAYER_NORM,
        mlp_ratio: int = 4,
        mlp_hidden_dim: int | None = None,
        ffn_dropout_rate: float = 0.0,
        weight_init_style: Literal["xavier_uniform", "pytorch", "fan_in_depth_scaled"] = "xavier_uniform",
        gpt_proj_rescale: bool = False,
        use_bias: bool = True,
        post_trunk_norm: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        assert not isinstance(self_attn_target, nn.Module), (
            "self_attn_target shouldn't be a nn.Module. Otherwise self_attn_target is shared across blocks!"
        )

        assert not isinstance(cross_attn_target, nn.Module), (
            "cross_attn_target shouldn't be a nn.Module. Otherwise cross_attn_target is shared across blocks!"
        )

        assert not isinstance(ffn_target, nn.Module), (
            "ffn_target should be a nn.Module. Otherwise ffn_target is shared across blocks!"
        )

        assert not isinstance(act_layer, nn.Module), (
            "act_layer should be a nn.Module. Otherwise act_layer is shared across blocks!"
        )

        assert not isinstance(norm_layer, nn.Module), (
            "norm_layer should be a nn.Module. Otherwise norm_layer is shared across blocks!"
        )

        self.encoder_output_key = encoder_output_key
        self.self_attn_mask_read_key = self_attn_mask_read_key
        self.gpt_proj_rescale = gpt_proj_rescale

        if drop_path_type == "progressive":
            dpr = list(np.linspace(0, drop_path_rate, num_blocks))
        elif drop_path_type == "uniform":
            dpr = [drop_path_rate for _ in range(num_blocks)]
        else:
            raise ValueError(f"drop_path_type needs to be 'progressive' or 'uniform' ({drop_path_type} provided).")

        mlp_hidden_dim = mlp_hidden_dim if mlp_hidden_dim is not None else int(mlp_ratio * embed_dim)

        self.blocks = nn.Sequential(
            *[
                block(
                    dim=embed_dim,
                    self_attn_target=self_attn_target,
                    cross_attn_target=cross_attn_target,
                    ffn_target=ffn_target,
                    mlp_hidden_dim=mlp_hidden_dim,
                    ffn_dropout_rate=ffn_dropout_rate,
                    norm_layer=norm_layer,
                    drop_path=dpr[i],
                    use_bias=use_bias,
                )
                for i in range(num_blocks)
            ]
        )

        self.post_trunk_norm = None
        if post_trunk_norm:
            self.post_trunk_norm = norm_layer(embed_dim)

        self.weight_init_style = weight_init_style

    def init_weights(self) -> None:
        self.apply(self._init_weights)
        if self.gpt_proj_rescale:
            self._gpt_proj_rescale()

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            if self.weight_init_style == "xavier_uniform":
                # Based on MAE and official Jax ViT implementation
                torch.nn.init.xavier_uniform_(m.weight)
            elif self.weight_init_style == "pytorch":
                # PyTorch ViT uses trunc_normal_
                trunc_normal_(m.weight, std=0.02)
            elif self.weight_init_style == "fan_in_depth_scaled":
                a = 1 / (1 + len(self.blocks))
                variance_scaling_(m.weight, scale=a, mode="fan_in", distribution="truncated_normal")
            elif self.weight_init_style is not None:
                raise ValueError(f"Got an unsupported `weight_init_style` -> {self.weight_init_style}.")

            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _gpt_proj_rescale(self) -> None:
        # apply special scaled init to the residual projections, per GPT-2 paper
        def rescale(param: torch.Tensor, depth: int) -> None:
            param.div_(math.sqrt(2.0 * depth))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, data_dict: DATA_DICT, **_: Any) -> DATA_DICT:
        tokens = data_dict[self.read_key]
        encoder_outputs = data_dict[self.encoder_output_key]
        mask = data_dict.get(self.self_attn_mask_read_key, None)

        features = []
        for blk in self.blocks:
            tokens = blk(tokens, encoder_outputs=encoder_outputs, mask=mask)
            features.append(tokens)

        if self.post_trunk_norm is not None:
            tokens = self.post_trunk_norm(tokens)

        data_dict[self.write_key] = tokens
        data_dict[f"{self.write_key}_decoder"] = features

        return data_dict


class TPWrapper(nn.Module):
    """Tensor Parallelism wrapper around the Transformer Decoder.

    TP is picky with how you structure the nn.Module's forward and cannot handle the data_dict properly.
    So, we basically wrap the Transformer Decoder to do the process of reading and writing from the data_dict
    outside of the actual forward.

    Args:
        decoder: transformer decoder model to be wrapped.
    """

    def __init__(self, decoder: TransformerDecoder):
        super().__init__()

        self.decoder = decoder

        # get the keys directly from the transformer
        self.read_key = self.decoder.read_key
        self.encoder_output_key = self.decoder.encoder_output_key
        self.self_attn_mask_read_key = self.decoder.self_attn_mask_read_key
        self.write_key = self.decoder.write_key

        # monkey patch the transformer forward to remove data_dict stuff
        def _forward(
            self,
            tokens: torch.Tensor,
            encoder_outputs: torch.Tensor,
            mask: torch.Tensor | None,
        ) -> tuple[torch.Tensor, list[torch.Tensor]]:
            features = []
            for blk in self.blocks:
                tokens = blk(tokens, encoder_outputs=encoder_outputs, mask=mask)
                features.append(tokens)

            if self.post_trunk_norm is not None:
                tokens = self.post_trunk_norm(tokens)

            return tokens, features

        self.decoder.forward = MethodType(_forward, self.decoder)

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        tokens = data_dict[self.read_key]
        encoder_outputs = data_dict[self.encoder_output_key]
        mask = data_dict.get(self.self_attn_mask_read_key, None)

        tokens, features = self.transformer(tokens, encoder_outputs, mask)

        data_dict[self.write_key] = tokens
        data_dict[f"{self.write_key}_decoder"] = features

        return data_dict
