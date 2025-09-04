# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import os
import pathlib
import pickle
from collections.abc import Callable
from typing import Any

import numpy as np

from l3m.data.vision.datasets.image_folder import CustomImageFolder

__all__ = [
    "ImageNet",
    "ImageNetA",
    "ImageNetR",
    "ImageNetSketch",
    "ImageNetV2Top",
    "ImageNetV2Thr",
    "ImageNetV2MatchedFreq",
    "ImageNetWithTextLabel",
]


class ImageNet(CustomImageFolder):
    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        classes = list(_load_classes()["inet"])
        mapping = dict(zip(classes, range(len(classes)), strict=True))
        return classes, mapping


class ImageNetA(CustomImageFolder):
    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        classes = list(_load_classes()["inet-a"])
        mapping = dict(zip(classes, range(len(classes)), strict=True))
        return classes, mapping


class ImageNetR(CustomImageFolder):
    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        classes = list(_load_classes()["inet-r"])
        mapping = dict(zip(classes, range(len(classes)), strict=True))
        return classes, mapping


class ImageNetSketch(CustomImageFolder):
    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        classes = list(_load_classes()["inet-sketch"])
        mapping = dict(zip(classes, range(len(classes)), strict=True))
        return classes, mapping


class ImageNetV2Top(CustomImageFolder):
    def __init__(self, root: str, *args: Any, **kwargs: Any):
        root = os.path.join(root, "imagenetv2-topimages", "validation")
        super().__init__(root, *args, **kwargs)


class ImageNetV2Thr(CustomImageFolder):
    def __init__(self, root: str, *args: Any, **kwargs: Any):
        root = os.path.join(root, "imagenetv2-threshold0.7", "validation")
        super().__init__(root, *args, **kwargs)


class ImageNetV2MatchedFreq(CustomImageFolder):
    def __init__(self, root: str, *args: Any, **kwargs: Any):
        root = os.path.join(root, "imagenetv2-matched-frequency", "validation")
        super().__init__(root, *args, **kwargs)

    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        classes = list(_load_classes()["inet-v2"])
        mapping = dict(zip(classes, range(len(classes)), strict=True))
        return classes, mapping


class ImageNetWithTextLabel(CustomImageFolder):
    def __init__(
        self,
        tokenizer: Callable[[Any], tuple[Any, ...]],
        imagenet_class_names_path: str,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.imagenet_class_names = np.load(imagenet_class_names_path, allow_pickle=True)

    def __getitem__(self, index: int) -> dict[str, Any]:
        data_dict = super().__getitem__(index)
        data_dict["text"] = self.tokenizer(self.imagenet_class_names[data_dict["target"]])[0]
        return data_dict


def _load_classes() -> dict[str, tuple[str, ...]]:
    root = pathlib.Path(__file__).parent.parent.parent.parent.parent.parent
    with open(root / "data/assets/imagenet_classes.pkl", "rb") as fin:
        return pickle.load(fin)
