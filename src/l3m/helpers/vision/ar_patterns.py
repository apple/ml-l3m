# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from functools import lru_cache

import torch

__all__ = [
    "get_spiral_permutation",
    "get_checkerboard_pattern",
    "get_fixed_random_pattern",
    "get_reverse_raster_pattern",
    "get_horizontal_raster_pattern",
    "get_reverse_horizontal_raster_pattern",
    "FIXED_RANDOM_PATTERNS",
]


@lru_cache
def get_spiral_permutation(grid_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a spiral permutation of indices for a given grid size.

    This function returns two permutations: one that arranges the indices in a spiral pattern,
    and its inverse.  The spiral starts from the outside and moves inwards, ending in the center.

    Args:
        grid_size: A tuple or list representing the height and width of the grid (H, W).

    Returns:
        A tuple containing the permutations and its inverse.
    """

    H, W = grid_size
    moves_x = [1, 0, -1, 0]
    moves_y = [0, 1, 0, -1]
    x, y = 0, 0
    direction = 0
    visited = torch.zeros(H, W).bool()
    mat = torch.arange(H * W).reshape(H, W)
    permutation = []

    for _ in range(H * W):
        permutation.append(mat[x, y].item())
        visited[x, y] = True
        new_x = x + moves_x[direction]
        new_y = y + moves_y[direction]

        # Do not change direction if within range and not visited, otherwise change
        if not (0 <= new_x < H and 0 <= new_y < W and not visited[new_x, new_y]):
            direction = (direction + 1) % 4
            new_x = x + moves_x[direction]
            new_y = y + moves_y[direction]

        x, y = new_x, new_y

    # Reverse the vector to start the spiral from the middle
    permutation = torch.LongTensor(permutation[::-1])
    inverse_permutation = torch.argsort(permutation)

    return permutation, inverse_permutation


@lru_cache
def get_checkerboard_pattern(grid_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a checkerboard pattern permutation of indices for a given grid size.

    This function returns two permutations: one that arranges the indices in a checkerboard pattern,
    and its inverse.

    Args:
        grid_size: A tuple or list representing the height and width of the grid (H, W).

    Returns:
        A tuple containing the permutations and its inverse.
    """

    H, W = grid_size
    mask = torch.zeros(H, W).bool()
    mask[1::2, 1::2] = True
    mask[::2, ::2] = True
    mat = torch.arange(H * W).reshape(H, W)
    permutation = torch.cat([mat[mask], mat[~mask]])
    inverse_permutation = torch.argsort(permutation)
    return permutation, inverse_permutation


@lru_cache
def get_fixed_random_pattern(grid_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Retrieves a pre-defined fixed random permutation of indices for a given grid size.

    This function returns two permutations: a fixed random permutation and its inverse.
    The fixed random patterns are pre-defined for specific grid sizes.

    Args:
        grid_size: A tuple or list representing the height and width of the grid (H, W).
            The product of H and W must be a key in the `FIXED_RANDOM_PATTERNS` dictionary.

    Returns:
        A tuple containing the permutations and its inverse.

    Raises:
        AssertionError: If the product of H and W is not a key in `FIXED_RANDOM_PATTERNS`.
    """

    H, W = grid_size
    N = H * W

    assert N in FIXED_RANDOM_PATTERNS, f"grid_size={grid_size} does not have a supported fixed random pattern."

    permutation = FIXED_RANDOM_PATTERNS[N]
    inverse_permutation = torch.argsort(permutation)
    return permutation, inverse_permutation


@lru_cache
def get_reverse_raster_pattern(
    grid_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a reverse raster scan permutation of indices for a given grid size.

    This function returns two permutations: one that arranges the indices in reverse raster scan order,
    and its inverse. Reverse raster scan means starting from the bottom-right corner and moving leftwards, then upwards.

    Args:
        grid_size: A tuple or list representing the height and width of the grid (H, W).

    Returns:
        A tuple containing the permutations and its inverse.
    """

    H, W = grid_size
    N = H * W
    permutation = torch.arange(N).flip(dims=(0,))
    inverse_permutation = torch.argsort(permutation)
    return permutation, inverse_permutation


@lru_cache
def get_horizontal_raster_pattern(
    grid_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a horizontal raster scan permutation of indices for a given grid size.

    This function returns two permutations: one that arranges the indices in a horizontal raster scan order,
    and its inverse.  Horizontal raster scan means traversing the grid row by row (horizontally).

    Args:
        grid_size: A tuple or list representing the height and width of the grid (H, W).

    Returns:
        A tuple containing the permutations and its inverse.
    """

    H, W = grid_size
    mat = torch.arange(H * W).reshape(H, W)
    permutation = mat.T.flatten()
    inverse_permutation = torch.argsort(permutation)
    return permutation, inverse_permutation


@lru_cache
def get_reverse_horizontal_raster_pattern(
    grid_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a reverse horizontal raster scan permutation of indices for a given grid size.

    This function returns two permutations: one that arranges the indices in a reverse horizontal raster scan order,
    and its inverse.
    Reverse horizontal raster scan means traversing the grid row by row (horizontally) in reverse order,
    starting from the last element.

    Args:
        grid_size: A tuple or list representing the height and width of the grid (H, W).

    Returns:
        A tuple containing the permutations and its inverse.
    """

    H, W = grid_size
    mat = torch.arange(H * W).reshape(H, W)
    permutation = mat.T.flatten().flip(dims=(0,))
    inverse_permutation = torch.argsort(permutation)
    return permutation, inverse_permutation


# fmt: off
FIXED_RANDOM_PATTERNS = {
    256:  torch.tensor([125,  30, 224, 253, 136, 184, 101, 133, 132, 181,  24, 209, 254, 206,
                        107,  31, 134, 169, 112,  81,  43, 127, 214, 177, 129,  15, 113, 250,
                        229, 111, 155,  73,  79, 228,   3,  46, 159,  59, 173,  39,  94,  42,
                        121, 202, 205, 213,  23, 117,  80,  13, 193, 139,  70, 200,  35,  20,
                        146,  85, 152,  87,  21,  74, 115,  65, 122,  58, 137, 168, 154, 234,
                        110, 157,  89, 199,  50, 241,  72,  48, 105, 118, 162, 164, 120, 126,
                        194, 175, 246,  99, 108, 188, 123, 192, 233,  51,  45, 217, 201, 151,
                        182,  78, 211, 109, 150, 176, 100, 243, 147, 174, 236,  19,  25,  36,
                        44,  47, 225, 138, 143,  97,  10, 207, 215,  11, 130, 255, 210, 103,
                        34, 222, 166,  52, 216,  29, 226, 251,  32,  17,  82, 160,   9, 179,
                        8,  53, 148, 149, 161,  77,  68,  27,  37, 185, 247,  93,  56, 135,
                        76, 178, 104,  40, 190, 158, 219, 204,  95, 249, 238,  54, 145,  62,
                        187,  96,  18, 218, 156, 197, 106, 116,   2, 171, 144, 220,  41, 223,
                        195,  84, 170, 186,  67, 131,  60,  49, 244, 240, 198,  61, 227, 141,
                        124, 242,  86, 142,  28,  22,  12, 212, 245,  98,  91,   0, 221, 208,
                        7,  69,  75, 140,  16,  26,  38, 191, 167, 119, 172, 165, 153, 196,
                        92, 183, 235,  63, 203, 230,  55, 231,  33,  71,  64, 248, 102,  90,
                        57,   4,  88, 239, 189, 114,   1, 232,  66, 252, 128,   5,  83, 237,
                        14, 163, 180,   6]),

    196: torch.tensor([49, 178, 138, 141,  56, 182,  50, 191,  79, 169, 174, 110, 183, 140,
                       132, 176,  37, 115,  97, 139,  88,  58,  52,  68, 101, 145,   0,  85,
                       16, 106, 163, 156,  39, 135,  90, 160, 180, 120,  64,  55,  22,  62,
                       32,  17, 185, 123, 137,  47, 181, 186,  89,  91, 128, 133, 148,  93,
                       42,  38, 113,  73,  63, 179, 118, 109, 171,  25,  98, 121, 194, 152,
                       149,  83, 119,  53, 104,  74, 190,  46,  20,  33, 124, 130, 159,  18,
                       167, 155, 192, 108,  57,  75,  45,  48, 117,   6,  71, 162, 164, 195,
                       173, 131,  99, 142, 158, 187,  29, 153,  23,  24,  11, 129, 143,  13,
                       134,   9,  66,  54, 102,  65,  41, 157, 150, 168, 154,  82,  44,  84,
                       170,  92,  51,  95,  86, 166, 147, 122,  21, 111, 107,   4,  30,   1,
                       69,  10, 116, 184, 151,  80, 193,  60,   7,  28,   8,  14,  70,  34,
                       146, 177,  81,  78,  19,  94,  26,  61,  59,  27,  31, 126,   5,  87,
                       127,  96,  77,  35, 100,   3, 165, 114,  43, 103, 189,  72,  15,  36,
                       188, 161, 112, 172, 105,  40, 136,   2,  67,  76, 175, 125,  12, 144])
}
# fmt: on
