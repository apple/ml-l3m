# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Any

import torch.nn as nn

from l3m.constants.generic import HPARAM_PREFIX
from l3m.constants.typing import DATA_DICT

__all__ = ["ProbeHparamSearchWrapper"]


class ProbeHparamSearchWrapper(nn.Module):
    """Wraps a head module to perform a hyperparameter search over specified grids.

    This does not actually assign the different hparams, but names the heads such that we can parse it afterwards.

    Args:
        head: The head module constructor.
        lr_grid: Grid of learning rates to search over.
        wd_grid: Grid of weight decays to search over.
        batchnorm_grid: Grid of batchnorm modes to search over.
    """

    def __init__(
        self,
        head: nn.Module,
        lr_grid: tuple[float, ...],
        wd_grid: tuple[float, ...] = (),
        batchnorm_grid: tuple[bool, ...] = (False,),
    ):
        super().__init__()

        self.heads = nn.ModuleDict()
        for lr in lr_grid:
            for bn_mode in batchnorm_grid:
                key = f"{HPARAM_PREFIX}_LR_{lr}_BN_{bn_mode}".replace(".", ",")
                if len(wd_grid):
                    for wd in wd_grid:
                        self.heads[f"{key}_WD_{wd}".replace(".", ",")] = head(use_batchnorm=bn_mode)
                else:
                    self.heads[key] = head(use_batchnorm=bn_mode)

    def forward(self, data_dict: DATA_DICT, **_: Any) -> dict[str, DATA_DICT]:
        output = {}
        for key, head in self.heads.items():
            output[key] = head(data_dict.copy())

        return output
