# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import itertools
import sys
from collections.abc import Iterator, Sized
from typing import Any

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

from l3m.helpers.dist.utils import DeviceMeshHandler

__all__ = ["InfiniteDataSampler"]


def _get_torch_dtype(size: int) -> torch.dtype:
    """Adapted from https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/samplers.py#L60"""
    return torch.int32 if size <= 2**31 else torch.int64


def _generate_randperm_indices(*, size: int, generator: torch.Generator) -> Iterator[torch.Tensor]:
    """Generate the indices of a random permutation.

    Adapted from: https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/samplers.py#L63
    """
    dtype = _get_torch_dtype(size)
    perm = torch.arange(size, dtype=dtype)
    for i in range(size):
        j = torch.randint(i, size, size=(1,), generator=generator).item()

        # Always swap even if no-op
        value = perm[j].item()
        perm[j] = perm[i].item()
        perm[i] = value
        yield value


class InfiniteDataSampler(Sampler):
    r"""Distributed Sampler that samples infinitely.

    It should be used with iteration based training loops. The iterator includes
    an infinite while loop to repeat the sampling every time yielding the generator
    finishes.

    .. note::
         Adapted from:

         - https://github.com/pytorch/pytorch/blob/main/torch/utils/data/distributed.py
         - https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/samplers.py#L165

    Args:
        dataset: Dataset used for sampling.
        num_replicas: Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank: Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed group.
        shuffle: If ``True`` (default), sampler will shuffle the indices.
        seed: Random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
    """

    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
        seek_ahead: int | None = 0,
    ):
        assert isinstance(dataset, Sized), type(dataset)
        super().__init__()
        if num_replicas is None or rank is None:
            rank_info = DeviceMeshHandler.get_rank_info()
            rank, num_replicas = rank_info.model_rank, rank_info.world_size

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

        self.num_samples = len(self.dataset) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

        self.shuffle = shuffle
        self.seed = seed

        self.seek_ahead = seek_ahead

    def __iter__(self) -> Iterator[Any]:
        if self.shuffle:
            iterator = self._shuffled_iterator()
        else:
            iterator = self._iterator()

        yield from itertools.islice(iterator, self.seek_ahead, None)

    def __len__(self) -> int:
        return sys.maxsize

    def _iterator(self) -> Iterator[Any]:
        while True:
            iterable = range(self.total_size)
            yield from itertools.islice(iterable, self.rank, None, self.num_replicas)

    def _shuffled_iterator(self) -> Iterator[Any]:
        generator = torch.Generator().manual_seed(self.seed)

        while True:
            iterable = _generate_randperm_indices(size=self.total_size, generator=generator)
            yield from itertools.islice(iterable, self.rank, None, self.num_replicas)
