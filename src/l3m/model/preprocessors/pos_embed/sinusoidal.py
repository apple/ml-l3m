# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""Adapted from MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py"""

import numpy as np

__all__ = ["get_2d_sincos_pos_embed", "get_3d_sincos_pos_embed"]


def get_3d_sincos_pos_embed(
    embed_dim: int,
    grid_size: tuple[int, int, int],
    cls_token: bool = False,
) -> np.ndarray:
    """Generates 3D sine-cosine positional embeddings.

    Args:
        embed_dim: Embedding dimension.
        grid_size: Size of the grid (depth, height, width).
        cls_token: Whether to include a class token.

    Returns:
        Positional embeddings.
    """

    grid_t = np.arange(grid_size[0], dtype=np.float32)
    grid_h = np.arange(grid_size[1], dtype=np.float32)
    grid_w = np.arange(grid_size[2], dtype=np.float32)
    grid_h, grid_t, grid_w = np.meshgrid(grid_h, grid_t, grid_w)  # (h,d,w) as the order
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid_t, grid_h, grid_w)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(
    embed_dim: int,
    grid_t: np.ndarray,
    grid_h: np.ndarray,
    grid_w: np.ndarray,
) -> np.ndarray:
    """Generates 3D sincos positional embeddings from grid dimensions.

    Args:
        embed_dim: Embedding dimension.
        grid_t: Grid in the time dimension.
        grid_h: Grid in the height dimension.
        grid_w: Grid in the width dimension.

    Returns:
         3D sincos positional embeddings.
    """
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_t)  # (H*W, D/2)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 4, grid_h)  # (H*W, D/4)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 4, grid_w)  # (H*W, D/4)
    emb = np.concatenate([emb_d, emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int | tuple[int, int],
    cls_token: bool = False,
) -> np.ndarray:
    """Generates 2D sine-cosine positional embeddings.

    Args:
        embed_dim: Embedding dimension.
        grid_size: Grid size (e.g., (H, W)).
        cls_token: Whether to prepend a class token.

    Returns:
        Positional embeddings.
    """

    grid_size = grid_size if isinstance(grid_size, tuple) else (grid_size, grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int,
    grid: np.ndarray,
) -> np.ndarray:
    """Generates 2D sine-cosine positional embeddings from a grid.

    Args:
        embed_dim: The dimension of the embedding. Must be even.
        grid: The dimensions of the grid (width, height).

    Returns:
        The 2D sine-cosine positional embeddings.
    """

    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_w, emb_h], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generates sin/cos position embedding from a set of positions.

    Args:
        embed_dim: Dimension of the embedding vector.
        pos: A torch tensor of positions.

    Returns:
        An array containing the positional embeddings.
    """

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
