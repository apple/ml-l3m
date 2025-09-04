# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Any

import numpy as np
import torch

from l3m.model.meta_models import ReadWriteClass

__all__ = [
    "RandomMasking",
    "RasterMasking",
    "RandomRasterMasking",
]


class RandomMasking(ReadWriteClass):
    """Creates a random mask for images, MAE style.

    Args:
        num_patches: Number of image patches.
        masking_ratio: Percentage of image patches that are masked.
    """

    def __init__(
        self,
        num_patches: int = 196,
        masking_ratio: float = 0.75,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.num_patches = num_patches
        self.masking_ratio = masking_ratio
        self.patches_to_keep = int(self.num_patches * (1 - self.masking_ratio))

    def __call__(self) -> torch.Tensor:
        noise = torch.rand(self.num_patches)

        ids_keep = torch.argsort(noise)
        ids_restore = torch.argsort(ids_keep)

        mask = torch.ones(self.num_patches)
        mask[: self.patches_to_keep] = 0
        mask = torch.gather(mask, dim=0, index=ids_restore)
        return mask.to(torch.bool)


class RasterMasking(ReadWriteClass):
    """Creates a causal mask for images.

    Args:
        num_patches: Number of image patches.
        masking_ratio: Percentage of image patches that are masked.
    """

    def __init__(
        self,
        num_patches: int = 196,
        masking_ratio: float = 0.5,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.num_patches = num_patches
        self.masking_ratio = masking_ratio
        self.patches_to_keep = int(self.num_patches * (1 - self.masking_ratio))

    def __call__(self) -> torch.Tensor:
        mask = torch.ones(self.num_patches)
        mask[: self.patches_to_keep] = 0
        return mask.to(torch.bool)


class RandomRasterMasking(ReadWriteClass):
    """Creates a prefix mask for images.

    Args:
        num_patches: Number of image patches.
        prefix_range: Range to sample prefix size from.
        sample_start: Whether to sample the start position for the prefix mask.
        force_full_attn: Whether to ignore and force bidirectional attention.
        seed: Random seed.
    """

    def __init__(
        self,
        num_patches: int = 256,
        prefix_range: tuple[int, int] = (1, 255),
        sample_start: bool = False,
        force_full_attn: bool = False,
        seed: int = 0,
        multipleof: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.num_patches = num_patches
        self.prefix_range = prefix_range
        self.sample_start = sample_start
        self.force_full_attn = force_full_attn
        self.multipleof = multipleof
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self) -> torch.Tensor:
        prefix_len_candidates = range(self.prefix_range[0], self.prefix_range[1] + 1)
        valid_prefix_lens = [x for x in prefix_len_candidates if x % self.multipleof == 0]
        prefix_len = self.rng.choice(valid_prefix_lens)

        mask = torch.ones(self.num_patches).to(torch.bool)

        if self.force_full_attn:
            return ~mask

        start_idx = 0
        if self.sample_start:
            start_idx = self.rng.integers(0, self.num_patches - prefix_len)

        mask[start_idx : start_idx + prefix_len] = False
        return mask
