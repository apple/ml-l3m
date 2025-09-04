# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from collections.abc import Callable, Iterable
from typing import Any

from l3m.constants.typing import DATA_DICT

from . import accuracy

__all__ = ["MetricsComputer", "accuracy"]


class MetricsComputer:
    """Thin wrapper to compute multiple metrics.

    Args:
        metrics: Metrics to compute.
    """

    def __init__(
        self,
        metrics: Iterable[Callable[[dict[str, Any], Any], dict[str, Any]]] = (),
        postprocessors: dict[str, dict] = None,
    ):
        self.metrics = tuple(metrics)
        self.postprocessors = postprocessors

    def __call__(self, data_dict: DATA_DICT, **kwargs: Any) -> DATA_DICT:
        """Compute the metrics.

        .. warning::
            No check is performed to ensure metric doesn't overwrite the keys
            of previously computed metrics.

        Args:
            data_dict: Input data passed to each metric.

        Returns:
            Dictionary containing the results.
        """
        stats = {}
        for metric in self.metrics:
            dataset_name = data_dict.get("dataset_name", None)
            if dataset_name is None or dataset_name in getattr(metric, "dataset_name", ()):
                stats.update(metric(data_dict, **kwargs))

        return stats

    def apply_postprocessing(self, stats: dict[str, Any], prefix: str) -> dict[str, Any]:
        if self.postprocessors is None:
            return {}

        postprocessed_stats = {}
        for write_key, postprocessor in self.postprocessors.items():
            postprocessing_fn = postprocessor.postprocess_fn
            pattern = postprocessor.pattern
            write_key = f"{prefix}_{write_key}"
            matched_values = [v for n, v in stats.items() if pattern in n]
            postprocessed_stats[write_key] = postprocessing_fn(matched_values)

        return postprocessed_stats
