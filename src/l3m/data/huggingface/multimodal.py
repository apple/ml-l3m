# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from collections.abc import Callable
from typing import Any

from datasets import load_dataset

from l3m.constants.typing import DATA_DICT

__all__ = ["SimpleImageTextDataset"]


class SimpleImageTextDataset:
    """Simple image-text dataset, replace with your actual dataset."""

    def __init__(
        self,
        transforms: Callable[[Any], Any],
        tokenizer: Callable[[str], tuple[Any, ...]],
        generators: list[Callable[[], Any]] = None,
    ):
        self.dataset = load_dataset("sezenkarakus/image-description-dataset-v2", split="train")
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.generators = generators

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> DATA_DICT:
        sample = self.dataset[index]
        image = sample["image"]
        text = sample["text"]

        # transform image and tokenize text
        image = self.transforms(image)
        text = self.tokenizer(text)[0]

        # construct sample
        sample = {"image": image, "text": text}

        # apply generators, e.g. the prefix mask creator
        if self.generators:
            for g in self.generators:
                sample[g.write_key] = g()

        sample["dataset_name"] = "image-description-dataset-v2"

        return sample
