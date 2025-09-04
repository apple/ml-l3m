# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import logging
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from l3m.constants.typing import DATA_DICT
from l3m.helpers.dist.utils import DeviceMeshHandler
from l3m.model.meta_models import ReadWriteBlock

__all__ = ["LLMCrossEntropyLoss", "MultiTokenLLMCrossEntropyLoss"]

logger = logging.getLogger("l3m")


def momentum_update(ema: float, new_value: float, alpha=0.95) -> float:
    return alpha * ema + (1 - alpha) * new_value


class LLMCrossEntropyLoss(ReadWriteBlock):
    """Computes the cross entropy loss for language models.

    Args:
        z_loss: Weight for the z-loss regularization term.
        shift_target: Whether to shift the target sequence to the left.
        ignore_index: Index to ignore when computing the loss.
        target_read_key: Key for the target in the data dictionary.
        renorm_non_padded: Whether to renormalize the loss based on the number of non-padded tokens.
        renorm_momentum: Whether to use momentum when renormalizing.
    """

    def __init__(
        self,
        z_loss: float = 0.0,
        shift_target: bool = True,
        ignore_index: int = -100,
        target_read_key: str = "text",
        renorm_non_padded: bool = False,
        renorm_momentum: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.renorm_non_padded = renorm_non_padded
        self.renorm_momentum = renorm_momentum
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.z_loss = z_loss
        self.shift_target = shift_target
        self.target_read_key = target_read_key

    @torch.no_grad()
    def compute_renorm_factor(self, target, avg_non_padded_local=None) -> tuple[float, float]:
        # Compute actual and expected number of non padded tokens locally
        non_padded = (target != self.ignore_index).sum()
        avg_non_padded = (avg_non_padded_local or non_padded).clone().float()

        # Gather expected number of non padded tokens from all processes
        dist.all_reduce(avg_non_padded, op=dist.ReduceOp.AVG)

        # Optionally use momentum to renormalize even more accurately across steps
        if self.renorm_momentum:
            if not hasattr(self, "avg_non_padded"):
                self.avg_non_padded = avg_non_padded.clone().float()
            avg_non_padded = momentum_update(self.avg_non_padded, avg_non_padded)
            self.avg_non_padded = avg_non_padded.clone()

        # Compute renorm factor
        renorm_factor = non_padded / avg_non_padded
        return renorm_factor.item(), avg_non_padded.item()

    def forward(self, data_dict: DATA_DICT, **_: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute the loss.

        Args:
            data_dict: Input data containing the following keys:

                - ``'{self.read_key}'`` - predictions of shape ``[B, L, D]``.
                - ``'{self.target_read_key}'`` - targets of shape ``[B, L]``.

        Returns:
            - sum of the cross entropy and the weighted z loss.
            - metrics with the following keys:

              - ``'perplexity'`` - perplexity.
              - ``'cross_entropy'`` - cross entropy loss.

              If :attr:`z_loss > 0 <z_loss>`, it also contains:

              - ``'z_loss'`` - Z loss.
              - ``'cross_entropy_with_z_loss'``
        """

        metrics = {}
        assert isinstance(self.read_key, str)
        assert self.read_key in data_dict, data_dict.keys()
        assert self.target_read_key in data_dict, data_dict.keys()

        preds, target = data_dict[self.read_key], data_dict[self.target_read_key]

        if preds.size(1) != target.size(1):
            assert not DeviceMeshHandler.cp_enabled(), (
                "With CP we can't do the shifting on local tensors as the seq dim is sharded."
            )
            if self.shift_target:
                # Shift target left, this assumes that the target is sequence
                # length is longer than `preds` by one element
                assert target.size(1) == preds.size(1) + 1, (
                    "If `shift_target=True`, the target is expected to be longer than predictions be one elements."
                )
                target = target[:, 1:]
            else:
                # This assumes a left padded input to the model, therefore the
                # predictions should be one element longer than the target
                assert target.size(1) == preds.size(1) - 1, (
                    "Expected predictions to be one element longer than target for `shift_target=False`"
                )
                preds = preds[:, :-1]

        target = target.flatten()
        # Force casting to FP32
        preds = preds.float().flatten(0, 1)

        loss = self.criterion(preds, target)

        metrics["perplexity"] = torch.exp(loss).item()
        metrics["cross_entropy"] = loss.item()

        # Renormalize according to the average number of non-padded tokens across all GPUs
        if self.renorm_non_padded and dist.is_initialized():
            renorm_factor, avg_non_padded = self.compute_renorm_factor(
                target=target, avg_non_padded_local=data_dict.get("avg_non_padded")
            )
            loss = loss * renorm_factor
            metrics["avg_non_padded"] = avg_non_padded

        if self.z_loss > 0.0:
            aux_loss = self.z_loss * preds.logsumexp(-1).pow(2).mean()
            loss = loss + aux_loss

            metrics["z_loss"] = aux_loss.item()
            metrics["cross_entropy_with_z_loss"] = loss.item()
        return loss, metrics


class MultiTokenLLMCrossEntropyLoss(LLMCrossEntropyLoss):
    """Computes the cross entropy loss for language models with multi-token prediction.

    The loss is computed by predicting multiple tokens at once, using different block sizes.

    Args:
        ignore_index: Index to ignore in the target.
        block_size: List of block sizes for multi-token prediction.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        block_size: list[int] = None,
        **kwargs: Any,
    ):
        super().__init__(ignore_index=ignore_index, **kwargs)

        if block_size is None:
            block_size = [1, 2, 3, 4]

        if not any(bs < 1 for bs in block_size):
            logger.warning("Block size < 1 found in the block size list.")

        self.block_size = block_size
        self.ignore_index = ignore_index

    def compute_loss(
        self,
        preds: torch.tensor,
        target: torch.tensor,
        block_size: int,
    ) -> dict[str, Any]:
        preds = preds[:, :-block_size]
        preds = preds.float().flatten(0, 1)

        target = target[:, block_size:]
        target = target.flatten()

        loss = self.criterion(preds, target)
        num_targets = (target != self.ignore_index).sum()

        return {"loss": loss, "num_targets": num_targets}

    def forward(self, data_dict: DATA_DICT, **_: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute the loss.

        Args:
            data_dict: Input data containing the following keys:

                - ``'{self.read_key}'`` - predictions of shape ``[B, L, D]``.
                - ``'text'`` - targets of shape ``[B, L]``.

        Returns:
            - sum of the cross entropy and the weighted z loss.
            - metrics with the following keys:

              - ``'perplexity'`` - perplexity.
              - ``'cross_entropy'`` - cross entropy loss.

              If :attr:`z_loss > 0 <z_loss>`, it also contains:

              - ``'z_loss'`` - Z loss.
              - ``'cross_entropy_with_z_loss'``
        """
        metrics = {}
        assert isinstance(self.read_key, str)
        preds, target = data_dict[self.read_key], data_dict[self.target_read_key]

        assert not DeviceMeshHandler.cp_enabled(), (
            "With CP we can't do the shifting on local tensors as the seq dim is sharded."
        )
        if self.shift_target:
            # Shift target left, this assumes that the target is sequence
            # length is longer than `preds` by one element
            assert target.size(1) == preds.size(1) + 1, (
                "If `shift_target=True`, the target is expected to be longer than predictions be one elements."
            )
            target = target[:, 1:]
        else:
            # This assumes a left padded input to the model, therefore the
            # predictions should be one element longer than the target
            assert target.size(1) == preds.size(1) - 1, (
                "Expected predictions to be one element longer than target for `shift_target=False`"
            )
            preds = preds[:, :-1]

        # make sure we have preds for all the heads
        assert len(self.block_size) == preds.size(2)

        # multi-token prediction
        loss_dicts = [self.compute_loss(preds[:, :, head], target, bs) for head, bs in enumerate(self.block_size)]

        # weight loss according to the number of targets
        total_targets = sum([d["num_targets"] for d in loss_dicts])
        loss = sum([d["loss"] * d["num_targets"] for d in loss_dicts]) / total_targets

        metrics["perplexity"] = torch.exp(loss).item()
        metrics["cross_entropy"] = loss.item()

        if self.z_loss > 0.0:
            aux_loss = self.z_loss * preds.logsumexp(-1).pow(2).mean()
            loss = loss + aux_loss

            metrics["z_loss"] = aux_loss.item()
            metrics["cross_entropy_with_z_loss"] = loss.item()

        return loss, metrics
