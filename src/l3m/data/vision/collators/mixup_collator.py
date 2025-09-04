# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from collections.abc import Callable
from typing import Any

from torch.utils.data import default_collate

__all__ = ["MixupCollator"]


class MixupCollator:
    def __init__(self, mixup_fn: Callable):
        self.mixup_fn = mixup_fn

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        assert all(key in batch[0] for key in ["image", "target"]), (
            "`MixupCollator` expects `image` and `target` keys in batch."
        )

        collated_batch = default_collate(batch)

        collated_batch["image"], collated_batch["target"] = self.mixup_fn(
            collated_batch["image"], collated_batch["target"]
        )
        return collated_batch
