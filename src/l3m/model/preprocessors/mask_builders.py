# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import and_masks, create_block_mask

from l3m.constants.typing import DATA_DICT
from l3m.model.meta_models import ReadWriteBlock

__all__ = [
    "MaybeCausalAttentionMaskBuilder",
    "PrefixAttentionMaskBuilder",
    "FlexAttnPrefixAttentionMaskBuilder",
    "MultimodalDecoderMaskBuilder",
    "PaddingAttentionMaskBuilder",
    "DocumentMaskingBuilder",
]


logger = logging.getLogger("l3m")


class MaybeCausalAttentionMaskBuilder(ReadWriteBlock):
    """Text Preprocessor.

    This module converts a sequence of text tokens into a sequence of embeddings, adds a class token (optional),
    and incorporates positional embeddings (optional).  It prepares text data for input to a transformer model.

    Args:
        vocab_size: The size of the vocabulary.
        embed_dim: The dimension of the token embeddings.
        context_length: The maximum length of the input sequence. Defaults to 77.
        cls_token: Whether to prepend a class token to the sequence. Defaults to True.
        pos_embed_type: The type of positional embedding to use (absolute or sincos). Defaults to "absolute".
        inference_read_key: The key to use for reading text data during inference. Defaults to "prompt".
        init_style: The initialization style for the embeddings. Defaults to "normal".
        norm_layer: Normalization layer to apply. Defaults to None.
        cls_token_init: How to initialize the cls_token. Defaults to "normal".
    """

    def __init__(self, seq_len: int, cls_token: bool = True, **kwargs: Any):
        super().__init__(**kwargs)

        seq_len = seq_len + int(cls_token)
        self.register_buffer(
            "causal_mask",
            torch.ones(1, seq_len, seq_len, dtype=torch.bool).tril(diagonal=0),
            persistent=False,
        )
        self.register_buffer(
            "non_causal_mask",
            torch.ones(1, seq_len, seq_len, dtype=torch.bool),
            persistent=False,
        )

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        selector = data_dict[self.read_key]

        B = selector.size(0)
        selector = selector.view(B, 1, 1)
        causal_masks = self.causal_mask.expand(B, -1, -1)
        non_causal_masks = self.non_causal_mask.expand(B, -1, -1)
        attn_masks = selector * causal_masks + (1 - selector) * non_causal_masks

        data_dict[self.write_key] = attn_masks.bool()
        return data_dict


class PrefixAttentionMaskBuilder(ReadWriteBlock):
    """Prefix Attention Mask Builder.

    This module constructs an attention mask that combines a causal mask with a prefix mask. It's intended for use
    in models where some tokens attend to all previous tokens (causal attention), while other tokens (the prefix) attend
    to all tokens in the sequence.

    Args:
        seq_len: The length of the sequence.
        cls_token: Whether a class token is included in the sequence.
        num_registers: The number of register tokens appended to the sequence.
    """

    def __init__(
        self,
        seq_len: int,
        cls_token: bool = False,
        num_registers: int = 0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.num_registers = num_registers
        seq_len = seq_len + self.num_registers + int(cls_token)
        self.seq_len = seq_len

        self.register_buffer(
            "attn_mask",
            torch.empty(1, seq_len, seq_len, dtype=torch.bool),
            persistent=False,
        )

        self.init_weights()

    def init_weights(self) -> None:
        attn_mask = torch.ones(1, self.seq_len, self.seq_len, dtype=torch.bool).tril(diagonal=0)

        if self.num_registers:
            logger.warning("The use of registers is only supported with prefix attention when append_side='right'.")
            attn_mask[:, -self.num_registers :, :] = True

        self.attn_mask.data.copy_(attn_mask)

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        mask = data_dict[self.read_key]

        if self.num_registers:
            mask = F.pad(mask, (0, self.num_registers), value=1)

        B, N = mask.size()
        prefix_mask = (~mask).unsqueeze(1).expand(-1, N, -1).bool()

        attn_mask = self.attn_mask.clone().expand(B, -1, -1)
        attn_mask = torch.logical_or(attn_mask, prefix_mask)

        data_dict[self.write_key] = attn_mask
        return data_dict


class FlexAttnPrefixAttentionMaskBuilder(ReadWriteBlock):
    """Creates prefix mask for FlexAttention.

    Args:
        read_key: Key in data_dict to read the mask from.
        write_key: Key in data_dict to write the modified mask to.
    """

    def forward(self, data_dict: DATA_DICT, **_) -> DATA_DICT:
        mask = data_dict[self.read_key]

        B, N = mask.shape[:2]
        prefix_length = N - mask.count_nonzero(dim=1)

        def prefix_mask(b, h, q_idx, kv_idx):
            return (kv_idx < prefix_length[b]) | (q_idx >= kv_idx)

        # mask is different per sequence so we set B equal to our batch size
        mask_mod = self.create_block_mask_cached(prefix_mask, B, None, N, N, device=mask.device)
        data_dict[self.write_key] = mask_mod

        return data_dict

    @lru_cache(maxsize=1024)  # noqa: B019
    @torch.compile
    def create_block_mask_cached(
        self,
        mask_mod: Callable,
        B: int,
        H: int,
        Q_LEN: int,
        KV_LEN: int,
        device: str | torch.device | int = "cuda",
        BLOCK_SIZE: int = 256,
    ) -> Any:
        return create_block_mask(
            mask_mod,
            B=B,
            H=H,
            Q_LEN=Q_LEN,
            KV_LEN=KV_LEN,
            device=device,
            BLOCK_SIZE=BLOCK_SIZE,
            _compile=True,
        )


class MultimodalDecoderMaskBuilder(ReadWriteBlock):
    """Creates a multimodal decoder mask.

    This mask builder assumes a sequence of (image_patches || text_tokens).

    Args:
        num_patches: Number of image patches.
        num_text_tokens: Number of text tokens.
        force_causal_mask: Whether to force a causal mask, ignoring prefix mask.
        read_key: Key in data_dict to read the mask from.
        write_key: Key in data_dict to write the modified mask to.
    """

    def __init__(
        self,
        num_patches: int,
        num_text_tokens: int,
        force_causal_mask: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.num_patches = num_patches
        self.num_text_tokens = num_text_tokens
        self.force_causal_mask = force_causal_mask

        seq_len = num_patches + num_text_tokens
        attn_mask = torch.ones(1, seq_len, seq_len, dtype=torch.bool).tril(diagonal=0)
        logger.warning("This Mask Builder assumes a sequence of (image_patches || text_tokens).")

        self.register_buffer(
            "attn_mask",
            attn_mask,
            persistent=False,
        )

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        mask = data_dict[self.read_key]
        if self.force_causal_mask:
            # Ignore Prefix mask, keep everything causal
            B = mask.size(0)
            attn_mask = self.attn_mask.clone().expand(B, -1, -1)
        else:
            mask = F.pad(mask, (0, self.num_text_tokens), value=1)
            B, N = mask.size()
            prefix_mask = (~mask).unsqueeze(1).expand(-1, N, -1).bool()
            attn_mask = self.attn_mask.clone().expand(B, -1, -1)
            attn_mask = torch.logical_or(attn_mask, prefix_mask)

        data_dict[self.write_key] = attn_mask
        return data_dict


class PaddingAttentionMaskBuilder(ReadWriteBlock):
    """Builds an attention mask from padding tokens."""

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        mask = data_dict[self.read_key]
        B, N = mask.size()

        # NOTE(alaaeldin_ali): The working example below assumes a padding_side=left,
        # it works equally for padding_side=right
        # 0 0 1 1 -> 0 0 0 0
        #            0 0 0 0
        #            1 1 1 1
        #            1 1 1 1
        expand_right = (~mask).unsqueeze(-1).expand(-1, -1, N).bool()

        # 0 0 1 1 -> 0 0 1 1
        #            0 0 1 1
        #            0 0 1 1
        #            0 0 1 1
        expand_down = (~mask).unsqueeze(1).expand(-1, N, -1).bool()

        # 0 0 0 0
        # 0 0 0 0
        # 0 0 1 1
        # 0 0 1 1
        intersection = torch.logical_and(expand_down, expand_right)

        # NOTE(alaaeldin_ali): Inverse union is important to avoid having the pad rows with all
        # -inf attn weights (pre-softmax)
        # 1 1 0 0
        # 1 1 0 0
        # 0 0 0 0
        # 0 0 0 0
        inverse_union = ~torch.logical_or(expand_down, expand_right)

        # 1 1 0 0
        # 1 1 0 0
        # 0 0 1 1
        # 0 0 1 1
        padding_mask = torch.logical_or(inverse_union, intersection)

        causal_mask = torch.ones(B, N, N, dtype=torch.bool).tril(diagonal=0)
        causal_mask = causal_mask.to(mask.device)

        attn_mask = torch.logical_and(causal_mask, padding_mask)
        data_dict[self.write_key] = attn_mask
        return data_dict


class DocumentMaskingBuilder(ReadWriteBlock):
    def forward(self, data_dict: DATA_DICT, **_: Any) -> DATA_DICT:
        document_id = data_dict[self.read_key]

        def document_masking(b, h, q_idx, kv_idx):
            return document_id[b, q_idx] == document_id[b, kv_idx]

        def causal_masking(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        mask_mod = and_masks(document_masking, causal_masking)

        B, N = document_id.shape
        block_mask = create_block_mask(
            mask_mod,
            B=B,
            H=None,  # broadcast along head dimension.
            Q_LEN=N,
            KV_LEN=N,
        )

        data_dict[self.write_key] = block_mask
        return data_dict
