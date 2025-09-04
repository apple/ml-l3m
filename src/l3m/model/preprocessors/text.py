# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from collections.abc import Callable
from typing import Any, Literal

import torch
import torch.nn as nn

from l3m.constants.typing import DATA_DICT
from l3m.helpers.dist.utils import DeviceMeshHandler, tensor_to_dtensor
from l3m.model.meta_models import ReadWriteBlock
from l3m.model.preprocessors.pos_embed import sinusoidal

__all__ = ["TextPreprocessor"]


class TextPreprocessor(ReadWriteBlock):
    """Transforms raw text data into embeddings.

    This module embeds text tokens, adds positional embeddings (if configured),
    and applies normalization. It's designed to preprocess text data for use in
    downstream models, particularly transformers.

    Args:
        embed_dim: Dimension of the token embeddings.
        vocab_size: Size of the vocabulary.
        context_length: Maximum sequence length.
        pos_embed_type: Type of positional embedding ("sincos" or "absolute").
        init_style: Initialization style for embeddings.
        norm_layer: Normalization layer to apply.
        inference_read_key: Key to read input during inference.
        drop_last: Whether to drop the last token in the sequence during training.
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        context_length: int,
        pos_embed_type: Literal["sincos", "absolute"] | None = None,
        init_style: Literal["normal", "sparse_transformer", "uniform"] = "uniform",
        norm_layer: Callable = None,
        inference_read_key: str = "prompt",
        drop_last: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.context_length = context_length
        self.pos_embed_type = pos_embed_type
        if self.pos_embed_type is not None:
            self.pos_embed = nn.Parameter(
                torch.empty(
                    1,
                    self.context_length,
                    self.embed_dim,
                ),
                requires_grad=(self.pos_embed_type == "absolute"),
            )
        else:
            self.pos_embed = None
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.init_style = init_style
        self.inference_read_key = inference_read_key
        self.drop_last = drop_last

        self.init_weights()

    def init_weights(self) -> None:
        if self.pos_embed_type == "absolute":
            torch.nn.init.normal_(self.pos_embed, std=0.02)
        elif self.pos_embed_type == "sincos":
            sincos_init = sinusoidal.get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                (self.context_length, 1),
            )
            # create tensor and convert it to dtensor if needed
            sincos_init = tensor_to_dtensor(torch.from_numpy(sincos_init).float().unsqueeze(0), self.pos_embed)
            self.pos_embed.data.copy_(sincos_init)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Embedding):
            if self.init_style == "normal":
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif self.init_style == "sparse_transformer":
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.125 * (self.embed_dim**-0.5))
            elif self.init_style == "uniform":
                torch.nn.init.uniform_(m.weight, -0.1, 0.1)
            elif self.init_style is None:
                pass
            else:
                raise ValueError(f"Invalid initialization style {self.init_style}.")

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        if not self.generation_mode:
            x = data_dict[self.read_key]
        else:
            x = data_dict.get(self.inference_read_key, data_dict.get(self.read_key, None))

        if not self.generation_mode and self.drop_last:
            assert not DeviceMeshHandler.cp_enabled(), (
                "With CP we can't do the drop_last on local tensors as the seq dim is sharded."
            )
            x = x[:, :-1]  # Drop last token
            if "position_ids" in data_dict:
                data_dict["position_ids"] = data_dict["position_ids"][:, :-1]
            if "document_mask" in data_dict:
                data_dict["document_mask"] = data_dict["document_mask"][:, :-1]

        B, N = x.shape
        x = self.embedding(x)
        x = self.norm(x)

        if self.pos_embed is not None:
            x = x + self.pos_embed[:, :N]

        data_dict[self.write_key] = x
        return data_dict
