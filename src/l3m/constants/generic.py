# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""Provides constants across scripts."""

import torch

__all__ = ["DTYPE_DICT", "DATASET_NAME_KEY", "HPARAM_PREFIX"]

DTYPE_DICT = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

DATASET_NAME_KEY = "dataset_name"
HPARAM_PREFIX = "HP"
