# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Any

import torch
import torch.nn.functional as F

from l3m.constants.typing import DATA_DICT
from l3m.helpers.dist import utils as dist_utils
from l3m.helpers.dist.gather import all_gather_batch, all_gather_batch_with_grad
from l3m.model.meta_models import ReadWriteBlock

__all__ = ["CLIPContrastiveLoss"]


class CLIPContrastiveLoss(ReadWriteBlock):
    """CLIP loss.

    Adapted from `SLIP <https://github.com/facebookresearch/SLIP/blob/main/losses.py>`_.

    Args:
        image_read_key: Key where images are stored.
        text_read_key: Key where text is stored.
        gather_with_grad: Whether to use gather across devices keeping the gradients.
    """

    def __init__(
        self,
        image_read_key: str,
        text_read_key: str,
        gather_with_grad: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.image_read_key = image_read_key
        self.text_read_key = text_read_key
        self.labels: torch.Tensor | None = None
        self.last_local_batch_size: int | None = None
        self.gather_fn = all_gather_batch_with_grad if gather_with_grad else all_gather_batch

    def forward(self, data_dict: DATA_DICT, **_: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute the loss.

        Args:
            data_dict: Data.

        Returns:
            - CLIP loss.
            - metrics with the following keys:

              - ``'clip_loss'`` - CLIP loss.
              - ``'clip_acc'`` - accuracy in :math:`[0, 100]`.
              - ``'logit_scale'`` - logit scale (1 / temperature).
        """
        image_embed = data_dict[self.image_read_key]
        text_embed = data_dict[self.text_read_key]

        rank_info = dist_utils.DeviceMeshHandler.get_rank_info()
        rank = rank_info.model_rank

        B = image_embed.size(0)
        if B != self.last_local_batch_size:
            self.labels = B * rank + torch.arange(B, device=image_embed.device)
            self.last_local_batch_size = B

        # gather features from all GPUs
        image_embed_all, text_embed_all = self.gather_fn([image_embed, text_embed])

        # cosine similarity as logits
        # NOTE: this assumes that features are already normalized and temperature-scaled
        logits_per_image = image_embed @ text_embed_all.t()
        logits_per_text = text_embed @ image_embed_all.t()

        loss = (F.cross_entropy(logits_per_image, self.labels) + F.cross_entropy(logits_per_text, self.labels)) / 2.0

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / B

        return loss, {
            "clip_loss": loss,
            "clip_acc": acc,
            "logit_scale": data_dict["logit_scale"].item(),
        }
