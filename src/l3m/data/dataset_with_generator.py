# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from collections.abc import Callable, Mapping
from typing import Any

import torch.utils.data as data

__all__ = ["DatasetWithGenerator"]


class DatasetWithGenerator(data.Dataset):
    def __init__(
        self,
        base_dataset: Mapping[int, dict[str, Any]],
        generators: Callable[[], Any] | list[Callable[[], Any]],
    ):
        super().__init__()
        self.base_dataset = base_dataset

        if isinstance(generators, Callable):
            generators = [generators]
        self.generators = generators

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        data_dict = self.base_dataset[index]
        for g in self.generators:
            data_dict[g.write_key] = g()
        return data_dict
