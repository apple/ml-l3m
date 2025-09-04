# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from collections.abc import Callable

from datasets import load_dataset

from l3m.constants.typing import DATA_DICT

__all__ = ["SimpleTextDataset"]


class SimpleTextDataset:
    """Simple text dataset, replace with your actual dataset."""

    def __init__(
        self,
        tokenizer: Callable,
        generators: list[Callable] = None,
    ):
        self.dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
        self.tokenizer = tokenizer
        self.generators = generators

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> DATA_DICT:
        sample = self.dataset[index]
        text = sample["text"]

        # apply tokenizer
        text = self.tokenizer(text)[0]

        # construct sample
        sample = {"text": text, "dataset_name": "wikitext"}

        return sample
