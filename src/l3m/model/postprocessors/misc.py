# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Any, Literal

import torch
import torch.nn as nn

from l3m.constants.typing import DATA_DICT
from l3m.model.meta_models import ReadWriteBlock
from l3m.model.postprocessors.pool import GAP, ConcatPooling, WeightedAveragePooling
from l3m.model.postprocessors.select import ExtractCLS

__all__ = ["AverageLastNLayers"]

AGGREGATORS = {
    "GAP": GAP,
    "CLS": ExtractCLS,
    "WAP": WeightedAveragePooling,
    "CONCAT": ConcatPooling,
    None: nn.Identity,
}


class AverageLastNLayers(ReadWriteBlock):
    """Averages the last N layers of a model.

    Args:
        layers: List of layer indices to average.
        base_aggregator: Aggregator to use for each layer.
        reduce: Whether to reduce the features along dimension 1.
    """

    def __init__(
        self,
        layers: list[int],
        base_aggregator: Literal["GAP", "CLS", "WAP", "CONCAT"] | None = "GAP",
        reduce: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.layers = layers
        self.base_aggregator = AGGREGATORS[base_aggregator](read_key=self.read_key, write_key=self.read_key)
        self.reduce = reduce

    def forward(self, data_dict: DATA_DICT, **_: Any) -> DATA_DICT:
        layer_features = data_dict[f"{self.read_key}_trunk"]
        feats = torch.stack([layer_features[layer_id] for layer_id in self.layers], dim=-1).mean(dim=-1)

        if self.reduce:
            feats = feats.mean(dim=1)

        data_dict[self.write_key] = feats
        return data_dict
