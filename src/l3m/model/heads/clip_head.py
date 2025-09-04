# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from l3m.constants.typing import DATA_DICT
from l3m.model.meta_models import ReadWriteBlock

__all__ = ["CLIPHead"]


class CLIPHead(ReadWriteBlock):
    """
    CLIP Head module.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        use_bias: Whether to use bias in the linear layer.
        apply_l2_norm: Whether to apply L2 normalization to the output features.
        apply_temperature: Whether to apply temperature scaling to the output features.
        init_temperature: Initial temperature value.
        max_logit_scale: Maximum logit scale value.
        learnable_temperature: Whether the temperature is learnable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = False,
        apply_l2_norm: bool = True,
        apply_temperature: bool = False,
        init_temperature: float = 0.07,
        max_logit_scale: int = 100,
        learnable_temperature: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        if apply_temperature and not apply_l2_norm:
            raise AssertionError("Temperature only makes sense with l2 norm.")

        self.apply_l2_norm = apply_l2_norm
        self.apply_temperature = apply_temperature
        self.max_logit_scale = max_logit_scale
        self.init_temperature = init_temperature
        self.learnable_temperature = learnable_temperature
        if self.apply_temperature:
            self.log_logit_scale = nn.Parameter(
                torch.empty(1, dtype=torch.float32),
                requires_grad=learnable_temperature,
            )

        if self.learnable_temperature:
            self.register_buffer(
                "max_log_logit_scale",
                torch.empty([], dtype=torch.float32),
            )

        self.linear = nn.Linear(in_features, out_features, bias=use_bias)

        self.init_weights()

    def init_weights(self) -> None:
        if self.apply_temperature:
            self.log_logit_scale.data.fill_(np.log(1 / self.init_temperature))
        if self.learnable_temperature:
            self.max_log_logit_scale.data.fill_(np.log(self.max_logit_scale))

    @torch.no_grad()
    def clamp_logit_scale(self) -> None:
        if self.learnable_temperature:
            self.log_logit_scale.data.clamp_(0, self.max_log_logit_scale)

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        features = self.linear(data_dict[self.read_key])

        if self.apply_l2_norm:
            features = F.normalize(features, p=2, dim=-1)

        if self.apply_temperature:
            self.clamp_logit_scale()
            logit_scale = self.log_logit_scale.exp()
            features = logit_scale * features
            data_dict["logit_scale"] = logit_scale

        data_dict[self.write_key] = features
        return data_dict
