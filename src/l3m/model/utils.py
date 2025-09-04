# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from l3m.constants.typing import DATA_DICT
from l3m.helpers.dist.utils import DeviceMeshHandler
from l3m.model.meta_models import ReadWriteBlock, ReadWriteClass

__all__ = ["MultiBlock", "ShuffleBatchDim"]


class MultiBlock(nn.Module):
    """Wraps a list of multiple blocks into a single class and executes them in sequence.

    Args:
        blocks: List of Blocks to be executed in sequence.
    """

    def __init__(self, blocks: list[ReadWriteClass] | list[ReadWriteBlock]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        for block in self.blocks:
            data_dict = block(data_dict)
        return data_dict


class ShuffleBatchDim(ReadWriteBlock):
    """Randomly shuffles the batch dimension. Used to corrupt the model's target.

    Args:
        shuffling_probability: Probability of applying the shuffle.
        seed: Seed for reproducibility.
    """

    def __init__(self, shuffling_probability: float, seed: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.shuffling_probability = shuffling_probability
        assert 0.0 < shuffling_probability < 1.0

        model_rank = DeviceMeshHandler.get_rank_info().model_rank
        self.rng = np.random.default_rng(seed=seed + model_rank)

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        x = data_dict[self.read_key]
        if self.shuffling_probability > self.rng.random():
            x = x[torch.randperm(x.size(0))]

        data_dict[self.write_key] = x
        return data_dict
