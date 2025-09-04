# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import torch
import torch.nn as nn

__all__ = ["RoPE"]


class RoPE(nn.Module):
    """HuggingFace implementation of Rotary Positional Embedding (RoPE).

    Args:
        embed_dim: The dimension of the embedding.
        num_heads: The number of attention heads.
        context_length: The maximum sequence length.
        theta: The scaling factor for the frequency.
    """

    def __init__(
        self,
        context_length: int,
        embed_dim: int | None = None,
        num_heads: int | None = None,
        head_dim: int | None = None,
        theta: float = 10000.0,
    ):
        super().__init__()

        assert embed_dim or head_dim, "Provide either embed_dim or head_dim."
        assert not (embed_dim and head_dim), "Both embed_dim and head_dim were provided, provide only one."

        if embed_dim:
            assert num_heads, "Provide num_heads when providing embed_dim."
            self.dim = embed_dim // num_heads
        elif head_dim:
            self.dim = head_dim

        self.max_position_embeddings = context_length
        self.base = theta

        self.register_buffer("inv_freq", torch.empty(int(self.dim / 2)), persistent=False)
        self.init_weights()

    def init_weights(self) -> None:
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.inv_freq.data.copy_(inv_freq)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_dim: int = 2,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if position_ids is None:
            position_ids = torch.arange(v.size(seq_dim)).unsqueeze(0).expand(v.size(0), -1)
            position_ids = position_ids.to(v.device)

        # x: [bs, num_attention_heads, seq_len, head_size]
        # [bs, dim/2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # [bs, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = v.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # [bs, seq_len, dim/2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # [bs, seq_len, dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Unsqueeze the heads dim
        # if seq_dim=2 (i.e. [B, H, N, D]) -> unsqueeze_dim=1  (default of this codebase)
        # if seq_dim=1 (i.e. [B, N, H, D]) -> unsqueeze_dim=2
        unsqueeze_dim = 1 if seq_dim == 2 else 2

        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        cos, sin = cos.to(dtype=v.dtype), sin.to(dtype=v.dtype)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed, v
