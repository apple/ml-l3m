# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""Provides unified typing across scripts."""

from collections.abc import Callable
from typing import Any

import torch

__all__ = ["DATA_DICT", "CHECKPOINT", "BASE_CRITERION", "CRITERION_WRAPPER"]

DATA_DICT = dict[str, Any]
CHECKPOINT = dict[str, Any]
BASE_CRITERION = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
CRITERION_WRAPPER = Callable[[dict[str, torch.Tensor]], tuple[torch.Tensor, dict[str, torch.Tensor]]]
