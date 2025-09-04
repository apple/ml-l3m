# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Any

import torch
import torch.nn as nn

from l3m.constants.typing import DATA_DICT
from l3m.model.meta_models import ReadWriteBlock

__all__ = ["RasterDiscreteLoss"]


class RasterDiscreteLoss(ReadWriteBlock):
    """Computes the cross entropy loss for discrete targets.

    The input ``data_dict`` is expected to contain keys for predictions, targets, and optionally a mask.

    Args:
        block_size: Size of the block for next patch prediction.
        target_read_key: Key for the target in the data dictionary.
        mask_read_key: Key for the mask in the data dictionary.
    """

    def __init__(
        self,
        block_size: int = 1,
        target_read_key: str = "image",
        mask_read_key: str = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.target_read_key = target_read_key
        self.mask_read_key = mask_read_key

    def forward(
        self,
        data_dict: DATA_DICT,
        **_: Any,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the loss.

        Args:
            data_dict: Input data containing the following keys:

                - ``'{self.read_key}'`` - predictions of shape ``[B, L, D]``.
                - ``'image'`` - targets of shape ``[B, L]``.

                Optionally, can contain the following keys:

                - ``'mask'`` - mask for the patches of shape ``[B, L]``.

        Returns:
            - discrete loss
            - metrics with the following keys:

              - ``'img_perplexity'`` - perplexity.
              - ``'img_cross_entropy_loss'`` - cross entropy loss.
        """
        assert self.target_read_key in data_dict, data_dict.keys()
        assert isinstance(self.read_key, str), type(self.read_key)
        metrics: dict[str, float | torch.Tensor] = {}

        pred, target, mask = (
            data_dict[self.read_key],
            data_dict[self.target_read_key],
            data_dict.get(self.mask_read_key, None),
        )
        B, N, D = pred.shape
        pred = pred.to(torch.float32)

        # next patch prediction
        if self.block_size == 0:
            if mask is not None:
                target = target[mask]
            else:
                target = target.flatten()

            pred = pred.flatten(0, 1)
        else:
            target = target[:, self.block_size :].flatten()
            pred = pred[:, : -self.block_size].flatten(0, 1)

        loss = nn.CrossEntropyLoss(reduce=False)(pred, target)
        if mask is not None:
            loss = loss.reshape(B, N - 1)
            mask = mask[:, self.block_size :]
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        else:
            loss = loss.mean()

        perplexity = torch.exp(loss)
        metrics["img_perplexity"] = perplexity.item()
        metrics["img_cross_entropy_loss"] = loss.item()

        return loss, metrics
