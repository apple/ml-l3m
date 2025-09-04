# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Any

import torch
import torch.nn as nn

from l3m.constants.typing import BASE_CRITERION, CRITERION_WRAPPER, DATA_DICT
from l3m.model.meta_models import ReadWriteBlock

__all__ = ["LossWrapper", "MultiHeadLossWrapper", "MultiLossWrapper"]


class LossWrapper(ReadWriteBlock):
    """Thin wrapper around a criterion.

    Args:
        base_criterion: base criterion.
        target_read_key: target key in the data dict.
    """

    def __init__(
        self,
        base_criterion: BASE_CRITERION,
        target_read_key: str = "target",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.base_criterion = base_criterion
        self.target_read_key = target_read_key

    def forward(self, data_dict: DATA_DICT, **_: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute the criterion.

        Args:
            data_dict: Input data containing the following keys:

                - ``'{self.read_key}'`` - the 1st argument for the :attr:`base_criterion`.
                - ``'target'`` - the 2nd argument for the :attr:`base_criterion`.
        Returns:
            - criterion.
            - metrics with the following keys:

              - ``'loss'`` - the criterion.
        """

        assert isinstance(self.read_key, str)
        assert self.target_read_key in data_dict

        pred, target = data_dict[self.read_key], data_dict[self.target_read_key]
        loss = self.base_criterion(pred, target)

        return loss, {"loss": loss.item()}


class MultiHeadLossWrapper(ReadWriteBlock):
    """Thin wrapper around a criterion to handle multiple predictions.

    Args:
        base_criterion: base criterion.
        target_read_key: target key in the data dict.
    """

    def __init__(
        self,
        base_criterion: BASE_CRITERION,
        target_read_key: str = "target",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.base_criterion = base_criterion
        self.target_read_key = target_read_key

    def forward(
        self,
        data_dict: DATA_DICT,
        **_: Any,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the criterion.

        Args:
            data_dict: Data containing the predictions.

        Returns:
            - sum of all individual criteria.
            - metrics with the following keys:

              - ``'{key}_loss'`` - criterion for each key in ``prediction``.
        """
        losses = {k: self.base_criterion(v[self.read_key], v[self.target_read_key]) for k, v in data_dict.items()}
        metrics = {f"{k}_loss": v.item() for k, v in losses.items()}

        return sum(losses.values()), metrics


class MultiLossWrapper(nn.Module):
    """Thin wrapper around multiple criterion wrappers.

    Args:
        criteria: loss criteria to wrap.
        criteria_weights: weights for each criterion.
    """

    def __init__(
        self,
        criteria: list[CRITERION_WRAPPER],
        criteria_weights: tuple[float, ...] = (1.0,),
    ):
        super().__init__()
        assert len(criteria) == len(criteria_weights), "Criteria and their corresponding weights do not match in count."
        self.criteria = criteria
        self.criteria_weights = criteria_weights

    def forward(
        self,
        data_dict: DATA_DICT,
        model: nn.Module | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute the criterion.

        Args:
            data_dict: Input data passed to each criterion in :attr:`criteria`.
            model: Model passed to the criterion.

        Returns:
            - the sum of all criteria.
            - metrics containing the metadata from each criterion.
        """
        all_metrics = {}
        total_loss: float | torch.Tensor = 0.0
        for criterion, criterion_weight in zip(self.criteria, self.criteria_weights, strict=False):
            loss, metrics = criterion(data_dict, model=model)
            total_loss = total_loss + (criterion_weight * loss)
            all_metrics.update(metrics)

        return total_loss, all_metrics
