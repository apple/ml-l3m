# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from collections.abc import Callable
from typing import Any

import numpy as np
from torchvision import datasets

from l3m.constants.typing import DATA_DICT

__all__ = [
    "Cifar10Dataset",
    "Cifar100Dataset",
    "Food101Dataset",
    "DTDDataset",
    "Cifar10DatasetWithTextLabel",
    "Cifar100DatasetWithTextLabel",
    "Food101DatasetWithTextLabel",
    "DTDDatasetWithTextLabel",
]


class Cifar10Dataset(datasets.CIFAR10):
    def __init__(self, *args: Any, **kwargs: Any):
        kwargs["train"] = kwargs.pop("split") == "train"
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> DATA_DICT:
        img, lab = super().__getitem__(index)
        return {"image": img, "target": lab}


class Cifar100Dataset(datasets.CIFAR100):
    def __init__(self, *args: Any, **kwargs: Any):
        kwargs["train"] = kwargs.pop("split") == "train"
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> DATA_DICT:
        img, lab = super().__getitem__(index)
        return {"image": img, "target": lab}


class Food101Dataset(datasets.Food101):
    def __getitem__(self, index: int) -> DATA_DICT:
        img, lab = super().__getitem__(index)
        return {"image": img, "target": lab}


class DTDDataset:
    def __init__(self, *args: Any, split: str = "test", **kwargs: Any):
        self.dataset = datasets.DTD(*args, split=split, **kwargs)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> DATA_DICT:
        img, lab = self.dataset[index]
        return {"image": img, "target": lab}


class WithTextLabelMixin:
    def __init__(
        self,
        *args: Any,
        tokenizer: Callable[[Any], tuple[Any, ...]],
        class_names_path: str,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.class_names = np.load(class_names_path, allow_pickle=True)

    def __getitem__(self, index: int) -> DATA_DICT:
        data_dict = super().__getitem__(index)
        data_dict["text"] = self.tokenizer(self.class_names[data_dict["target"]])[0]
        return data_dict


class Cifar10DatasetWithTextLabel(WithTextLabelMixin, Cifar10Dataset):
    pass


class Cifar100DatasetWithTextLabel(WithTextLabelMixin, Cifar100Dataset):
    pass


class Food101DatasetWithTextLabel(WithTextLabelMixin, Food101Dataset):
    pass


class DTDDatasetWithTextLabel(WithTextLabelMixin, DTDDataset):
    pass
