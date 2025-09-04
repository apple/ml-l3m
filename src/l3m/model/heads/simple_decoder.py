# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""Adapted from https://github.com/facebookresearch/mae/blob/main/models_mae.py"""

from collections.abc import Callable
from typing import Any, Literal

import torch
import torch.nn as nn
from timm.layers.weight_init import variance_scaling_

from l3m.constants.typing import DATA_DICT
from l3m.helpers.dist.utils import tensor_to_dtensor
from l3m.model.layers.ffn import MLP
from l3m.model.layers.normalization import LAYER_NORM
from l3m.model.meta_models import ReadWriteBlock
from l3m.model.preprocessors.pos_embed import sinusoidal
from l3m.model.trunks.transformer import Block

__all__ = ["SimpleDecoder", "DiscreteDecoder"]


class SimpleDecoder(ReadWriteBlock):
    """Simple decoder module.

    This module projects an input tensor to an embedding dimension,
    applies a series of transformer blocks, and predicts the output.

    Args:
        attn_target: Attention mechanism to use within each decoder block.
        ffn_target: Feedforward network to use within each decoder block.
        input_dim: Input dimension.
        num_layers: Number of decoder blocks.
        embed_dim: Embedding dimension.
        output_dim: Output dimension.
        mlp_ratio: Ratio of hidden dimension to embedding dimension.
        mlp_hidden_dim: Hidden dimension of the feedforward network.
        patch_size: Patch size for positional embeddings.
        image_size: Image size for positional embeddings.
        norm_layer: Normalization layer to use.
        logit_proj_init: Initialization method for the prediction layer.
        insert_mask: Whether to insert mask tokens.
        disable_pos_embed: Whether to disable positional embeddings.
        use_bias: Whether to use bias in linear layers.
        patch_dropout_rate: Probability of dropping patches.
        skip_projection: Whether to skip the initial projection layer.
        post_decoder_norm: Whether to apply normalization after the decoder blocks.
    """

    def __init__(
        self,
        attn_target: Callable[..., nn.Module],
        ffn_target: Callable[..., nn.Module] = MLP,
        input_dim: int = 1024,
        num_layers: int = 6,
        embed_dim: int = 512,
        output_dim: int = 768,
        mlp_ratio: int = 4,
        mlp_hidden_dim: int | None = None,
        patch_size: int = 16,
        image_size: int | tuple[int, int] = 224,
        norm_layer: Callable[[int], nn.Module] = LAYER_NORM,
        logit_proj_init: Literal["zero", "fan_in_depth_scaled"] = "fan_in_depth_scaled",
        insert_mask: bool = False,
        disable_pos_embed: bool = False,
        use_bias: bool = True,
        patch_dropout_rate: float = 0.0,
        skip_projection: bool = False,
        post_decoder_norm: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.patch_size, self.embed_dim, self.num_patches = (
            patch_size,
            embed_dim,
            (image_size[0] // patch_size) * (image_size[1] // patch_size),
        )
        self.grid_size = (image_size[0] // patch_size, image_size[1] // patch_size)

        if skip_projection:
            assert input_dim == embed_dim, (
                f"`input_dim` and `embed_dim` do not match {input_dim} != {embed_dim}. Cannot use `skip_projection`"
            )

        self.decoder_projection = nn.Linear(input_dim, embed_dim, bias=True) if not skip_projection else nn.Identity()

        if not disable_pos_embed:
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)

        mlp_hidden_dim = mlp_hidden_dim if mlp_hidden_dim is not None else int(mlp_ratio * embed_dim)

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    attn_target=attn_target,
                    ffn_target=ffn_target,
                    embed_dim=embed_dim,
                    mlp_hidden_dim=mlp_hidden_dim,
                    norm_layer=norm_layer,
                    use_bias=use_bias,
                )
                for _ in range(num_layers)
            ]
        )

        self.decoder_norm = norm_layer(embed_dim) if post_decoder_norm else nn.Identity()
        self.decoder_pred = nn.Linear(embed_dim, output_dim, bias=True)

        self.logit_proj_init = logit_proj_init
        self.disable_pos_embed = disable_pos_embed
        self.insert_mask = insert_mask
        self.patch_dropout_rate = patch_dropout_rate

        if insert_mask or patch_dropout_rate > 0.0:
            self.mask_token = nn.Parameter(torch.empty(1, 1, embed_dim))

        self.init_weights()

    def init_weights(self) -> None:
        if self.insert_mask or self.patch_dropout_rate > 0.0:
            self.mask_token.data.fill_(0.0)

        if self.patch_dropout_rate > 0.0:
            self.register_buffer(
                "patch_drop_bernoulli",
                torch.ones(1, self.num_patches) * self.patch_dropout_rate,
            )

        if not self.disable_pos_embed:
            decoder_pos_embed = sinusoidal.get_2d_sincos_pos_embed(self.embed_dim, self.grid_size, cls_token=False)
            # create tensor and convert it to dtensor if needed
            decoder_pos_embed = tensor_to_dtensor(
                torch.from_numpy(decoder_pos_embed).float().unsqueeze(0),
                self.decoder_pos_embed,
            )
            self.decoder_pos_embed.data.copy_(decoder_pos_embed)

        if self.logit_proj_init == "zero":
            torch.nn.init.constant_(self.decoder_pred.weight, 0)
        elif self.logit_proj_init == "fan_in_depth_scaled":
            a = 1 / (1 + len(self.decoder_blocks))
            variance_scaling_(
                self.decoder_pred.weight,
                scale=a,
                mode="fan_in",
                distribution="truncated_normal",
            )

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def insert_mask_tokens(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, _, D = x.shape
        N = mask.shape[-1]
        tmp = torch.empty(B, N, D).to(x.device).to(x.dtype)
        tmp[mask] = self.mask_token.to(x.dtype)
        tmp[~mask] = x.reshape(-1, D)
        x = tmp
        return x

    def _random_patch_dropout(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        patch_drop_bernoulli = self.patch_drop_bernoulli.expand(B, -1)
        drop_mask = torch.bernoulli(patch_drop_bernoulli).bool()
        x[drop_mask] = self.mask_token.to(x.dtype)
        return x

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        x = data_dict[self.read_key]
        mask = data_dict.get("mask", None)

        x = self.decoder_projection(x)

        if mask is not None and self.insert_mask:
            x = self.insert_mask_tokens(x, mask)

        if self.patch_dropout_rate > 0.0:
            x = self._random_patch_dropout(x)

        B, N, D = x.shape
        if not self.disable_pos_embed:
            x = x + self.decoder_pos_embed[:, :N]

        for _, blk in enumerate(self.decoder_blocks):
            x = blk(x, mask=mask)

        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        data_dict[self.write_key] = x

        return data_dict


class DiscreteDecoder(SimpleDecoder):
    """Discrete Version of the SimpleDecoder.


    Args:
        tokenizer: Pre-trained image tokenizer.
        tokenizer_img_size: Target image size for the tokenizer.
        image_key: Image key in the data_dict to read/write to.
    """

    def __init__(
        self,
        tokenizer: Callable[[int], nn.Module],
        tokenizer_img_size: int = 112,
        image_key: str = "image",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.image_key = image_key

        self.tokenizer = tokenizer(img_size=tokenizer_img_size)
        self.tokenizer.freeze()
        self.tokenizer.eval()

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        data_dict = super().forward(data_dict)

        target = data_dict[self.image_key]
        with torch.no_grad():
            target = self.tokenizer(target).reshape(target.shape[0], -1)

        # Overwrite the image pixels with its tokenized version
        data_dict[self.image_key] = target

        return data_dict
