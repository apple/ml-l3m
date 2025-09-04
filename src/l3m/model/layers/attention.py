# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from l3m.model.preprocessors.pos_embed import rope

__all__ = [
    "GenericAttention",
    "Attention",
    "EfficientAttention",
    "PrefixCausalAttention",
    "GeneralizedAttention",
    "AttentionWithMask",
    "FlexAttention",
    "HFAttention",
    "HFAttentionWithMask",
    "DCLMAttention",
]


class GenericAttention(nn.Module):
    """General self-attention layer. Has support for causal or user-provided masks.
    Allows qk_norm, different number of heads for the kv and RoPE.

    Args:
        dim: The input dimension.
        head_dim: The specific dimension per head.
        num_heads: The number of attention heads.
        qkv_bias: Whether to include bias in the query, key, and value projections.
        qk_scale: Optional scaling factor for the query-key dot product.
        attn_drop: Dropout probability for the attention weights.
        proj_drop: Dropout probability for the output projection.
        use_bias: Whether to include bias in the output projection.
        is_causal: Whether the attention is causal (e.g., for autoregressive models).
        num_kv_heads: The number of key and value heads. If None, defaults to num_heads.
        rope: Optional RoPE (Rotary Position Embedding) module.
        qk_norm: Optional normalization module for queries and keys.
        use_flex_attention: Whether to pre-compile flex-attention, strongly recommended if using flex-attention.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int | None = None,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_bias: bool = True,
        is_causal: bool = False,
        num_kv_heads: int | None = None,
        rope: nn.Module | None = None,
        qk_norm: nn.Module | None = None,
        use_flex_attention: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        if not head_dim:
            head_dim = dim // num_heads

        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(dim, head_dim * self.num_heads, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, head_dim * self.num_kv_heads, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, head_dim * self.num_kv_heads, bias=qkv_bias)

        self.attn_drop = attn_drop
        self.proj = nn.Linear(head_dim * self.num_heads, dim, bias=use_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = qk_norm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = qk_norm(head_dim) if qk_norm else nn.Identity()

        self.is_causal = is_causal
        self.rope = rope

        self.attention_fn = None
        if use_flex_attention:
            # make sure flex_attention is always compiled
            self.attention_fn = torch.compile(
                flex_attention,
                mode="max-autotune-no-cudagraphs",
            )

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        **_,
    ) -> torch.Tensor:
        B, N, C = x.shape

        if mask is not None:
            assert not self.is_causal, "Mask was provided but the attention is causal."
            if not isinstance(mask, BlockMask):
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        q = self.q_proj(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_kv_heads, -1).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_kv_heads, -1).permute(0, 2, 1, 3)

        k = self.repeat_kv(k, self.num_kv_groups)
        v = self.repeat_kv(v, self.num_kv_groups)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q, k, v = self.rope(q, k, v, position_ids=position_ids)

        # for flex attention
        if isinstance(mask, BlockMask):
            assert self.attention_fn is not None, "BlockMask was given but use_flex_attention was not set to True."
            x = self.attention_fn(q, k, v, block_mask=mask)
        else:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=self.is_causal,
                attn_mask=mask,
                dropout_p=self.attn_drop,
            )

        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version,
        # can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=use_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, **_: Any) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EfficientAttention(Attention):
    """Adapted from https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/attention.py"""

    def __init__(
        self,
        *args: Any,
        is_causal: bool = False,
        rope_pos_embed: nn.Module | None = None,
        qk_norm: nn.Module | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.is_causal = is_causal
        self.rope_pos_embed = rope_pos_embed

        head_dim = kwargs["dim"] // kwargs["num_heads"]
        self.q_norm = qk_norm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = qk_norm(head_dim) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, **_: Any) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope_pos_embed is not None:
            q, k, v = self.rope_pos_embed(q, k, v)

        x = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal, scale=self.scale)

        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PrefixCausalAttention(Attention):
    def __init__(self, *args: Any, is_causal: bool = False, num_patches: int = 256, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.is_causal = is_causal
        self.register_buffer(
            "attn_mask",
            torch.ones(1, num_patches, num_patches, dtype=torch.bool),
            persistent=False,
        )

    def init_weights(self) -> None:
        self.attn_mask.data.fill_(1).tril_(diagonal=0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, **_: Any) -> torch.Tensor:
        assert mask is not None, "A mask is required for PrefixLM Causal Attention"
        B, N, C = x.shape
        prefix_mask = (~mask).unsqueeze(1).expand(-1, N, -1).bool()
        attn_mask = self.attn_mask.clone().expand(B, -1, -1)
        attn_mask = torch.logical_or(attn_mask, prefix_mask)
        attn_mask = attn_mask.unsqueeze(1)  # (B, 1, N, N)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionWithMask(EfficientAttention):
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        **_: Any,
    ):
        assert mask is not None, "A mask is required for AttentionWithMask"
        B, N, C = x.shape

        mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope_pos_embed is not None:
            q, k, v = self.rope_pos_embed(q, k, v)

        x = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)

        x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlexAttentionWithMask(Attention):
    def __init__(
        self,
        *args: Any,
        rope_pos_embed: nn.Module | None = None,
        qk_norm: nn.Module | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.rope_pos_embed = rope_pos_embed

        head_dim = kwargs["dim"] // kwargs["num_heads"]
        self.q_norm = qk_norm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = qk_norm(head_dim) if qk_norm else nn.Identity()

        # make sure flex_attention is always compiled
        self.flex_attention = torch.compile(
            flex_attention,
            mode="max-autotune-no-cudagraphs",
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: BlockMask,
        **_: Any,
    ) -> torch.Tensor:
        assert isinstance(mask, BlockMask), "Please use block_mask for this layer."
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope_pos_embed is not None:
            q, k, v = self.rope_pos_embed(q, k, v)

        x = self.flex_attention(q, k, v, block_mask=mask)

        x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GeneralizedAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        encoder_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_bias: bool = True,
        is_causal: bool = False,
        relative_pos_embed: nn.Module | None = None,
        qk_norm: nn.Module | None = None,
        **_: Any,
    ):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.Wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wk = nn.Linear(encoder_dim, dim, bias=qkv_bias)
        self.Wv = nn.Linear(encoder_dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=use_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.is_causal = is_causal
        self.relative_pos_embed = relative_pos_embed

        self.q_norm = qk_norm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = qk_norm(head_dim) if qk_norm else nn.Identity()

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        **_,
    ) -> torch.Tensor:
        B, N, C = queries.shape
        M = keys.shape[1]

        q = self.Wq(queries).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.Wk(keys).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.Wv(values).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.relative_pos_embed is not None:
            q, k, v = self.relative_pos_embed(q, k, v)

        x = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)

        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HFAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int | None = None,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_bias: bool = True,
        is_causal: bool = False,
        num_key_value_heads: int | None = None,
        rope_pos_embed: nn.Module | None = None,
        qk_norm: nn.Module | None = None,
        sdpa: bool = True,
        num_patches: int | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        if not head_dim:
            head_dim = dim // num_heads

        # NOTE scale factor was wrong in my original version,
        # can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(dim, head_dim * self.num_heads, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, head_dim * self.num_key_value_heads, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, head_dim * self.num_key_value_heads, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.o_proj = nn.Linear(head_dim * self.num_heads, dim, bias=use_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = qk_norm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = qk_norm(head_dim) if qk_norm else nn.Identity()

        self.is_causal = is_causal
        self.rope_pos_embed = rope_pos_embed
        self.sdpa = sdpa
        if not sdpa and self.is_causal:
            assert num_patches is not None, "For causal attention without SDPA, `num_patches` is a required argument."
            self.register_buffer(
                "attn_mask",
                torch.ones(1, num_patches, num_patches, dtype=torch.bool),
                persistent=False,
            )

    def init_weights(self) -> None:
        if not self.sdpa and self.is_causal:
            self.attn_mask.data.fill_(1).tril_(diagonal=0)

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_key_value_heads, -1).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_key_value_heads, -1).permute(0, 2, 1, 3)

        k = self.repeat_kv(k, self.num_key_value_groups)
        v = self.repeat_kv(v, self.num_key_value_groups)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope_pos_embed is not None:
            q, k, v = self.rope_pos_embed(q, k, v, position_ids=position_ids)

        if self.sdpa:
            x = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
            x = x.transpose(1, 2).reshape(B, N, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # (b, h, lq, lk)
            if self.is_causal:
                min_dtype = torch.finfo(attn.dtype).min
                attn = attn.masked_fill(self.attn_mask == 0, min_dtype)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.o_proj(x)
        x = self.proj_drop(x)
        return x


class HFAttentionWithMask(HFAttention):
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert mask is not None, "A mask is required for HFAttentionWithMask"
        B, N, C = x.shape

        mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        q = self.q_proj(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_key_value_heads, -1).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_key_value_heads, -1).permute(0, 2, 1, 3)

        k = self.repeat_kv(k, self.num_key_value_groups)
        v = self.repeat_kv(v, self.num_key_value_groups)

        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope_pos_embed is not None:
            q, k, v = self.rope_pos_embed(q, k, v, position_ids=position_ids)

        x = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.o_proj(x)
        x = self.proj_drop(x)
        return x


class DCLMAttention(Attention):
    """Adapted from https://github.com/mlfoundations/open_lm/blob/main/open_lm/model.py#L117
    Applies the QK-Norm over the embed-dim not the head-dim
    """

    def __init__(
        self,
        *args: Any,
        is_causal: bool = False,
        rope_pos_embed: nn.Module | None = None,
        qk_norm: nn.Module | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.is_causal = is_causal

        self.rope_pos_embed = rope_pos_embed
        if rope_pos_embed is not None:
            assert isinstance(rope_pos_embed, rope.RoPE), "This class supports only with RoPE positional encoding."

        self.head_dim = kwargs["dim"] // kwargs["num_heads"]
        self.q_norm = qk_norm(kwargs["dim"]) if qk_norm else nn.Identity()
        self.k_norm = qk_norm(kwargs["dim"]) if qk_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        B, N, C = x.shape

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim)

        if self.rope_pos_embed is not None:
            # NOTE: It is important to set `seq_dim=1` since the head and sequence dims are switched
            q, k, v = self.rope_pos_embed(q, k, v, position_ids=position_ids, seq_dim=1)

        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

        if mask is not None:
            x = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            x = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)

        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class FlexAttention(Attention):
    def __init__(
        self,
        *args: Any,
        rope_pos_embed: nn.Module | None = None,
        qk_norm: nn.Module | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.rope_pos_embed = rope_pos_embed

        head_dim = kwargs["dim"] // kwargs["num_heads"]
        self.q_norm = qk_norm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = qk_norm(head_dim) if qk_norm else nn.Identity()

        # make sure flex_attention is always compiled
        self.flex_attention = torch.compile(flex_attention, mode="max-autotune-no-cudagraphs")

    def forward(
        self,
        x: torch.Tensor,
        mask: BlockMask,
        position_ids: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        assert isinstance(mask, BlockMask), "Please use block_mask for this layer."
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope_pos_embed is not None:
            q, k, v = self.rope_pos_embed(q, k, v, position_ids=position_ids)

        x = self.flex_attention(q, k, v, block_mask=mask)

        x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
