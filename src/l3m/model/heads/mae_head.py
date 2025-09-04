# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from l3m.helpers.dist.utils import tensor_to_dtensor
from l3m.model.preprocessors.pos_embed import sinusoidal
from l3m.model.trunks.transformer import Block

__all__ = ["MAEDecoder"]


class MAEDecoder(nn.Module):
    """MAE decoder module.

    Takes encoded tokens, mask tokens, and mask information to reconstruct the
    original input.

    Adapted from https://github.com/facebookresearch/mae/blob/main/models_mae.py.

    Args:
        attn_target: Attention mechanism to use in the decoder blocks.
        input_dim: Input dimension of the encoded tokens.
        num_layers: Number of decoder layers.
        embed_dim: Embedding dimension of the decoder.
        output_dim: Output dimension of the reconstructed tokens.
        mlp_ratio: Ratio of the hidden dimension to the embedding dimension in the MLP layers.
        mlp_hidden_dim: Hidden dimension of the MLP layers. If None, it defaults to mlp_ratio * embed_dim.
        patch_size: Size of the input patches.
        image_size: Size of the input image.
    """

    def __init__(
        self,
        attn_target: Callable[..., nn.Module],
        input_dim: int = 1024,
        num_layers: int = 6,
        embed_dim: int = 512,
        output_dim: int = 768,
        mlp_ratio: int = 4,
        mlp_hidden_dim: int | None = None,
        patch_size: int = 16,
        image_size: int = 224,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.patch_size, self.embed_dim, self.num_patches = (
            patch_size,
            embed_dim,
            (image_size // patch_size) ** 2,
        )
        self.decoder_projection = nn.Linear(input_dim, embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.empty(1, self.num_patches + 1, embed_dim), requires_grad=False)
        self.mask_token = nn.Parameter(torch.empty(1, 1, embed_dim))

        mlp_hidden_dim = mlp_hidden_dim if mlp_hidden_dim is not None else int(mlp_ratio * embed_dim)

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    attn_target=attn_target,
                    embed_dim=embed_dim,
                    mlp_hidden_dim=mlp_hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.decoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.decoder_pred = nn.Linear(embed_dim, output_dim, bias=True)

        self.init_weights()

    def init_weights(self) -> None:
        decoder_pos_embed = sinusoidal.get_2d_sincos_pos_embed(
            self.embed_dim, int(self.num_patches**0.5), cls_token=True
        )
        # create tensor and convert it to dtensor if needed
        decoder_pos_embed = tensor_to_dtensor(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0),
            self.decoder_pos_embed,
        )
        self.decoder_pos_embed.data.copy_(decoder_pos_embed)

        torch.nn.init.normal_(self.mask_token, std=0.02)

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
        tmp[~mask] = x[:, 1:].reshape(-1, D)
        x = torch.cat([x[:, :1], tmp], dim=1)
        return x

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        x = self.decoder_projection(x)
        x = self.insert_mask_tokens(x, mask)
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        x = x[:, 1:, :]

        return x
