# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.utils import accuracy

from l3m.constants.prompts import SIMPLE_IMAGENET_TEMPLATES
from l3m.constants.typing import DATA_DICT
from l3m.model.meta_models import ReadWriteClass

__all__ = ["Accuracy", "CLIPAccuracy"]


class Accuracy(ReadWriteClass):
    """Top-k accuracy.

    Args:
         topk: Values for which to compute the top-k accuracy.
         hparam_pattern: Pattern used to match keys in the dictionary. To be used in
            conjunction with :class:`~l3m.model.heads.wrappers.ProbeHparamSearchWrapper`
            and :attr:`~l3m.constants.generic.HPARAM_PREFIX`.
         kwargs: Keyword arguments for :class:`~l3m.model.meta_models.ReadWriteClass`.
    """

    PREFIX = "top"

    def __init__(
        self,
        topk: tuple[int, ...] = (1, 5),
        hparam_pattern: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.topk = topk
        self._hparam_pattern = None if hparam_pattern is None else re.compile(hparam_pattern)

    def _get_preds(self, data_dict: DATA_DICT) -> torch.Tensor:
        assert self.read_key in data_dict, data_dict.keys()
        return data_dict[self.read_key]

    def _get_target(self, data_dict: DATA_DICT) -> torch.Tensor:
        assert "target" in data_dict, data_dict.keys()
        return data_dict["target"]

    def _compute_one(self, data_dict: DATA_DICT) -> dict[str, float]:
        preds = self._get_preds(data_dict)
        target = self._get_target(data_dict)
        stats = accuracy(preds, target, self.topk)
        stats = {f"{self.PREFIX}_{self.topk[i]}": acc for i, acc in enumerate(stats)}
        return stats

    def _compute_many(self, data_dict: DATA_DICT) -> dict[str, float]:
        stats = {}
        for key in data_dict:
            if self._hparam_pattern.match(key):
                for top_k, value in self._compute_one(data_dict[key]).items():
                    stats[f"{key}_{top_k}"] = value
        return stats

    def __call__(self, data_dict: DATA_DICT, **_: Any) -> dict[str, Any]:
        """Compute the metric.

        Args:
            data_dict: Input data containing the following keys:

                - ``'{self.read_key}'`` - inputs of shape ``[B, CLS]`` over which to
                  compute the :func:`~torch.topk` accuracy.
                - ``'target'`` - targets of shape ``[B]``.

        Returns:
            - metrics with the following keys.

              - ``'top_{k}'`` top-k accuracy in :math:`[0, 100]` for each k
                in :attr:`topk`.
        """
        assert isinstance(self.read_key, str), type(self.read_key)
        if self._hparam_pattern is None:
            return self._compute_one(data_dict)
        return self._compute_many(data_dict)


class CLIPAccuracy(Accuracy):
    """Top-k zero-shot accuracy for CLIP models.

    The metric pre-computes target embeddings based on textual templates for
    ImageNet-1k categories and measures accuracy against such targets in
    zero-shot manner.

    Args:
        imagenet_class_names_path: Path to pickle file containing the
            strings for 1000 imagenet categories.
        tokenizer: Text tokenizer Callable to convert raw text to discrete
            tokens.
        kwargs: Keyword arguments for :class:`~l3m.metrics.accuracy.Accuracy`.
    """

    PREFIX = "zero_shot_top"

    def __init__(
        self,
        imagenet_class_names_path: str,
        tokenizer: Callable[[list[str]], torch.Tensor],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.text_embeds = None

        imagenet_classes = np.load(imagenet_class_names_path, allow_pickle=True)
        tokenized_imagenet_classes = []
        for label in imagenet_classes.values():
            imagenet_class_templates = [t.format(label) for t in SIMPLE_IMAGENET_TEMPLATES]
            imagenet_class_templates = tokenizer(imagenet_class_templates)
            tokenized_imagenet_classes.append(imagenet_class_templates)
        self.tokenized_imagenet_classes = torch.stack(tokenized_imagenet_classes, dim=1)

    @torch.no_grad()
    def _precompute_class_embeds(
        self,
        model: nn.Module,
        device: torch.cuda.device,
    ) -> None:
        """Pre-compute classification from textual templates.

        Args:
            model: Model to be evaluated. Needs to be an instance
                :class:`~l3m.model.meta_models.MetaModel` or its wrapper.
            device: Device to which we move the classification targets.

        Returns:
            Nothing, just replaces :attr:`text_embeds`.
        """
        with torch.amp.autocast("cuda", dtype=torch.float32):
            T, C, D = self.tokenized_imagenet_classes.shape
            tokenized_imagenet_classes = self.tokenized_imagenet_classes.reshape(T * C, D)
            text_embeds = model(
                {"text": tokenized_imagenet_classes.to(device)},
                model_name="text_encoder",
            )["text_embed"]
            text_embeds = text_embeds.reshape(T, C, -1)
            self.text_embeds = F.normalize(F.normalize(text_embeds, dim=-1).mean(dim=0), dim=-1)

    @torch.no_grad()
    def _get_preds(self, data_dict: DATA_DICT) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=torch.float32):
            image_embeds = data_dict[self.read_key]
            preds = image_embeds @ self.text_embeds.t()
        return preds

    def __call__(
        self,
        data_dict: DATA_DICT,
        model: nn.Module,
        reset: bool,
        **_: Any,
    ) -> dict[str, Any]:
        """Compute the metric.

        Args:
            data_dict: Input data containing the following keys:

                - ``'{self.read_key}'`` - inputs of shape ``[B, CLS]`` over which to
                  compute the :func:`~torch.topk` accuracy.
            model: The model to be evaluated.
            reset: A flag that triggers pre-computation of targets from templates
                using the updated version of the mode.

        Returns:
            Metrics with the following keys:

            - ``'zero_shot_top_{k}'`` - zero shot top-k accuracy in :math:`[0, 100]` for each k in :attr:`topk`.
        """
        if reset:
            device = self._get_target(data_dict).device
            self._precompute_class_embeds(model, device=device)
        return super().__call__(data_dict)
