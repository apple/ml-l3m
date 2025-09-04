# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import logging
import math
from collections.abc import Callable, Iterable
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
from l3m.model.layers.peft import Adapter
from l3m.model.meta_models import ReadWriteBlock

__all__ = [
    "Block",
    "BlockWithAdapter",
    "Transformer",
    "TPWrapper",
    "NonMonolithicTransformerFFN",
]

logger = logging.getLogger("l3m")


class Block(nn.Module):
    """Transformer block consisting of attention and MLP layers.

    Args:
        embed_dim: Embedding dimension.
        attn_target: Attention module.
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
        embed_dim: int,
        attn_target: Callable[[bool], nn.Module],
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

        self.attn = attn_target(use_bias=use_bias)
        self.norm_1 = norm_layer(embed_dim)
        self.mlp = ffn_target(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=ffn_dropout_rate,
            use_bias=use_bias,
        )
        self.norm_2 = norm_layer(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm_1(x), mask=mask, position_ids=position_ids))
        x = x + self.drop_path(self.mlp(self.norm_2(x)))
        return x


class BlockWithAdapter(Block):
    """Transformer block with adapter layers.

    Args:
        args: Arguments passed to the parent Block class.
        adapter_hidden_dim: Hidden dimension of the adapter layers.
        kwargs: Keyword arguments passed to the parent Block class.
    """

    def __init__(self, *args: Any, adapter_hidden_dim: int = 32, **kwargs: Any):
        super().__init__(*args, **kwargs)

        embed_dim = kwargs["dim"]
        self.adapter_1 = Adapter(embed_dim=embed_dim, adapter_hidden_dim=adapter_hidden_dim)
        self.adapter_2 = Adapter(embed_dim=embed_dim, adapter_hidden_dim=adapter_hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, **_: Any) -> torch.Tensor:
        x = x + self.drop_path(self.adapter_1(self.attn(self.norm_1(x), mask=mask)))
        x = x + self.drop_path(self.adapter_2(self.mlp(self.norm_2(x))))
        return x


class Transformer(ReadWriteBlock):
    """Transformer encoder consisting of multiple transformer blocks.

    Args:
        embed_dim: Embedding dimension.
        attn_target: Attention module constructor.
        num_blocks: Number of transformer blocks.
        ffn_target: Feedforward network module constructor.
        act_layer: Activation layer.
        block: Transformer block constructor.
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
        embed_dim: int,
        attn_target: Callable[[bool], nn.Module],
        num_blocks: int,
        ffn_target: Callable[..., nn.Module] = MLP,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        block: Callable[..., nn.Module] = Block,
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

        assert isinstance(self.read_key, str)
        assert isinstance(self.write_key, str)

        assert not isinstance(attn_target, nn.Module), (
            "attn_target shouldn't be a nn.Module. Otherwise attn_target is shared across blocks!"
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

        if drop_path_type == "progressive":
            dpr = list(np.linspace(0, drop_path_rate, num_blocks))
        elif drop_path_type == "uniform":
            dpr = [drop_path_rate for _ in range(num_blocks)]
        else:
            raise ValueError(f"drop_path_type needs to be 'progressive' or 'uniform' ({drop_path_type} provided).")

        self.self_attn_mask_read_key = self_attn_mask_read_key

        mlp_hidden_dim = mlp_hidden_dim if mlp_hidden_dim is not None else int(mlp_ratio * embed_dim)

        self.blocks = self._build_blocks(
            block=block,
            num_blocks=num_blocks,
            embed_dim=embed_dim,
            attn_target=attn_target,
            ffn_target=ffn_target,
            mlp_hidden_dim=mlp_hidden_dim,
            ffn_dropout_rate=ffn_dropout_rate,
            dpr=dpr,
            act_layer=act_layer,
            norm_layer=norm_layer,
            use_bias=use_bias,
        )

        self.post_trunk_norm = None
        if post_trunk_norm:
            self.post_trunk_norm = norm_layer(embed_dim)

        self.weight_init_style = weight_init_style
        self.gpt_proj_rescale = gpt_proj_rescale

        self.init_weights()

    def _build_blocks(
        self,
        block: Callable[..., nn.Module],
        num_blocks: int,
        attn_target: Callable[..., nn.Module],
        embed_dim: int,
        dpr: list[float],
        ffn_target: Callable[..., nn.Module] = MLP,
        norm_layer: Callable[[int], nn.Module] = LAYER_NORM,
        mlp_hidden_dim: int | None = None,
        ffn_dropout_rate: float = 0.0,
        use_bias: bool = True,
        act_layer: Callable[[], nn.Module] = nn.GELU,
    ) -> nn.Sequential:
        assert isinstance(block, Callable)
        return nn.Sequential(
            *[
                block(
                    embed_dim=embed_dim,
                    attn_target=attn_target,
                    ffn_target=ffn_target,
                    act_layer=act_layer,
                    mlp_hidden_dim=mlp_hidden_dim,
                    ffn_dropout_rate=ffn_dropout_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    use_bias=use_bias,
                )
                for i in range(num_blocks)
            ]
        )

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
            elif self.weight_init_style == "normal":
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif self.weight_init_style == "fan_in_normal":
                variance_scaling_(m.weight, scale=1.0, mode="fan_in", distribution="normal")
            elif self.weight_init_style is not None:
                raise ValueError(f"Got an unsupported `weight_init_style` -> {self.weight_init_style}.")

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _gpt_proj_rescale(self) -> None:
        # apply special scaled init to the residual projections, per GPT-2 paper
        def rescale(param, depth):
            param.div_(math.sqrt(2.0 * depth))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        tokens = data_dict[self.read_key]
        mask = data_dict.get(self.self_attn_mask_read_key, None)
        position_ids = data_dict.get("position_ids", None)

        features = []
        for blk in self.blocks:
            tokens = blk(tokens, mask=mask, position_ids=position_ids)
            features.append(tokens)

        if self.post_trunk_norm is not None:
            tokens = self.post_trunk_norm(tokens)

        data_dict[self.write_key] = tokens
        data_dict[f"{self.write_key}_trunk"] = features

        return data_dict


class TPWrapper(nn.Module):
    """Tensor Parallelism wrapper around the Transformer.

    TP is picky with how you structure the nn.Module's forward and cannot handle the data_dict properly.
    So, we basically wrap the Transformer to do the process of reading and writing from the data_dict
    outside of the actual forward.

    Args:
        transformer: Transformer model to be wrapped.
    """

    def __init__(self, transformer: Transformer):
        super().__init__()

        self.transformer = transformer

        # get the keys directly from the transformer
        self.read_key = self.transformer.read_key
        self.self_attn_mask_read_key = self.transformer.self_attn_mask_read_key
        self.write_key = self.transformer.write_key

        # monkey patch the transformer forward to remove data_dict stuff
        def _forward(
            self,
            tokens: torch.Tensor,
            mask: torch.Tensor | None,
            position_ids: torch.Tensor | None,
        ) -> tuple[torch.Tensor, list[torch.Tensor]]:
            features = []
            for blk in self.blocks:
                tokens = blk(tokens, mask=mask, position_ids=position_ids)
                features.append(tokens)

            if self.post_trunk_norm is not None:
                tokens = self.post_trunk_norm(tokens)

            return tokens, features

        self.transformer.forward = MethodType(_forward, self.transformer)

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        tokens = data_dict[self.read_key]
        mask = data_dict.get(self.self_attn_mask_read_key, None)
        # Required for skipping pad position in RoPE
        position_ids = data_dict.get("position_ids", None)

        tokens, features = self.transformer(tokens, mask, position_ids)

        data_dict[self.write_key] = tokens
        data_dict[f"{self.write_key}_trunk"] = features

        return data_dict


class NonMonolithicTransformerFFN(Transformer):
    """A Transformer with asymmetrical blocks where each can have a different
    FFN definition (e.g., Alternating Dense and MoE blocks)

    Args:
        block: A list of dicts, each dict with keys/values:
            - ``'module'`` - :class:`Block` constructor **with an `ffn_target` already passed**.
            - ``'layers'`` - list of integers indicating the layers where this module should be placed.
        num_blocks: Number of transformer blocks.
        attn_target: Attention module.
        embed_dim: Embedding dimension.
        dpr: List of drop path rates for each block.
        norm_layer: Normalization layer.
        mlp_hidden_dim: Hidden dimension of the MLP.
        ffn_dropout_rate: Dropout rate for the feedforward network.
        use_bias: Whether to use bias in linear layers.
        act_layer: Activation layer.
    """

    def _build_blocks(
        self,
        block: list[dict[str, list[int] | nn.Module]],
        num_blocks: int,
        attn_target: Callable[..., nn.Module],
        embed_dim: int,
        dpr: list[float],
        norm_layer: Callable[[int], nn.Module] = LAYER_NORM,
        mlp_hidden_dim: int | None = None,
        ffn_dropout_rate: float = 0.0,
        use_bias: bool = True,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        **_: Any,
    ) -> nn.Sequential:
        assert isinstance(block, Iterable), type(block)
        layer_to_block_dict = {}
        for mapping in block:
            layers, blk = mapping["layers"], mapping["module"]
            for layer in layers:
                assert 0 <= layer < num_blocks, "layer id needs to be in [0, `num_blocks`]"
                layer_to_block_dict[layer] = blk

        assert len(layer_to_block_dict.keys()) == num_blocks

        block_list = []
        for i in range(num_blocks):
            block_list.append(
                layer_to_block_dict[i](
                    embed_dim=embed_dim,
                    attn_target=attn_target,
                    act_layer=act_layer,
                    mlp_hidden_dim=mlp_hidden_dim,
                    ffn_dropout_rate=ffn_dropout_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    use_bias=use_bias,
                )
            )

        return nn.Sequential(*block_list)
