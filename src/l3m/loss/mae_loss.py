# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import logging
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn.functional as F

from l3m.constants.typing import DATA_DICT
from l3m.model.meta_models import ReadWriteBlock

__all__ = [
    "MAELoss",
    "RasterPixelLoss",
    "MultiTokenRasterPixelLoss",
    "VideoRasterPixelLoss",
]

logger = logging.getLogger("l3m")


class MAELoss(ReadWriteBlock):
    """MAE loss.

    Args:
        patch_size: patch size.
        norm_pix_loss: whether to standardize the target to have 0 mean and 1 std.
        kwargs: kwargs for :class:`~l3m.model.meta_models.ReadWriteBlock`.
    """

    def __init__(
        self,
        patch_size: int = 16,
        norm_pix_loss: bool = True,
        target_read_key: str = "image",
        mask_read_key: str = "mask",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        self.target_read_key = target_read_key
        self.mask_read_key = mask_read_key

    def forward(
        self,
        data_dict: DATA_DICT,
        **_: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the loss.

        Args:
            data_dict: Input data containing the following keys:

                - ``'{self.read_key}'`` - predictions of shape ``[B, L, D]``.
                - ``'image'`` - target images of shape ``[B, 3, H, W]`` to be
                  :attr:`patchified <patchify>`.
                - ``'mask'`` - mask for the patches.

        Returns:
            - MAE loss.
            - metrics with the following keys:

              - ``'mae_loss'`` - MAE loss.
        """
        assert isinstance(self.read_key, str), type(self.read_key)

        metrics = {}
        output = data_dict[self.read_key]
        target, mask = data_dict[self.target_read_key], data_dict[self.mask_read_key]

        target = self.patchify(target)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5

        loss = (output - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        metrics["mae_loss"] = loss.item()
        return loss, metrics

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert images to patches.

        .. warning::
            This function assumes that the height and the width are the same.

        .. seealso::
            See :meth:`unpatchify` on how to unpatchify the patches.

        Args:
            imgs: Images of shape ``[B, 3, H, W]``.

        Returns:
            Patches of shape ``[B, H * W // (P * P), P * P * 3]``.
        """
        p = self.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patches back to the original images.

        .. warning::
            This function assumes that the height and the width are the same.

        .. seealso::
            See :meth:`patchify` on how to patchify the images.

        Args:
            x: Patches of shape ``[B, (H * W) // (P ** 2), (P ** 2) * 3]``.

        Returns:
            Original images of shape ``[B, 3, H, W]``.
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs


class RasterPixelLoss(ReadWriteBlock):
    """Image raster pixel loss.

    Args:
        patch_size: Patch size.
        image_size: Image size.
        norm_pix_loss: Whether to standardize the target to have 0 mean and 1 std.
        block_size: Offset used for the next patch prediction.
        autoregressive_pattern_getter: Autoregressive pattern as defined in
            :mod:`~l3m.helpers.vision.ar_pattern`.
        target_read_key: Key where target images are stored when calling :meth:`forward`.
        mask_read_key: Key where mask is stored when calling :meth:`forward`.
        kwargs: Keyword arguments for :class:`~l3m.model.meta_models.ReadWriteBlock`.
    """

    def __init__(
        self,
        patch_size: int = 16,
        image_size: int | tuple[int, int] = 224,
        norm_pix_loss: bool = True,
        norm_pix_dim: int | tuple[int, int] = -1,
        block_size: int = 1,
        autoregressive_pattern_getter: Callable[[], tuple[torch.Tensor, torch.Tensor]] = None,
        target_read_key: str = "image",
        mask_read_key: str | None = None,
        loss_fn: Callable[[torch.Tensor, torch.Tensor, Any], torch.Tensor] = F.mse_loss,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.grid_size = (image_size[0] // patch_size, image_size[1] // patch_size)
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        self.norm_pix_dim = tuple(norm_pix_dim) if isinstance(norm_pix_dim, Iterable) else norm_pix_dim
        if block_size < 1:
            logger.warning("Block size is < 1.")
        self.block_size = block_size
        self.target_read_key = target_read_key
        self.mask_read_key = mask_read_key
        self.loss_fn = loss_fn

        if autoregressive_pattern_getter is not None:
            permutation, _ = autoregressive_pattern_getter()
            self.register_buffer("permutation", permutation)
        else:
            self.permutation = None

    def compute_loss(
        self,
        output: torch.tensor,
        target: torch.tensor,
        block_size: int,
        mask: torch.tensor,
    ) -> dict[str, Any]:
        """Computes the loss for a single block size."""

        if self.norm_pix_loss:
            mean = target.mean(dim=self.norm_pix_dim, keepdim=True)
            var = target.var(dim=self.norm_pix_dim, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        if block_size != 0:
            target = target[:, block_size:]
            output = output[:, :-block_size]
        else:
            if mask is not None:
                B, _, C = target.shape
                target = target[mask].reshape(B, -1, C)

        loss = self.loss_fn(output, target, reduction="none")
        if mask is not None:
            loss = loss.mean(dim=-1)  # [B, N]
            mask = mask[:, block_size:]
            num_targets = int(mask.sum())
            loss = (loss * mask).sum() / num_targets  # mean loss on removed patches
        else:
            loss = loss.mean()
            num_targets = target.size(0) * target.size(1)
        return {"loss": loss, "num_targets": num_targets}

    def forward(self, data_dict: DATA_DICT, **_: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute the loss.

        Args:
            data_dict: Input data containing the following keys:

                - ``'{self.read_key}'`` - predictions of shape ``[B, L, D]``.
                - ``'{self.target_read_key}'`` - target images or videos to be
                  :attr:`patchified <patchify>` with ``[B, 3, H, W]`` and
                  ``[B, 3, T, H, W]`` shapes, respectively.

                Optionally, can contain the following keys:

                - ``'mask'`` - mask for the patches of shape ``[B, L]``.

        Returns:
            - MSE loss
            - metrics with the following keys:

              - ``'mse_norm_pix'`` - MSE loss.

        """
        assert self.target_read_key in data_dict, data_dict.keys()

        metrics = {}
        assert isinstance(self.read_key, str)
        assert isinstance(self.target_read_key, str)
        output, target = data_dict[self.read_key], data_dict[self.target_read_key]

        mask = None
        if self.mask_read_key is not None:
            assert self.mask_read_key in data_dict
            mask = data_dict[self.mask_read_key]

        if target.ndim > 3:
            target = self.patchify(target)

        if target.shape[1] != output.shape[1]:
            target = target[:, -output.shape[1] :]

        if self.permutation is not None:
            target = target[:, self.permutation]
            output = output[:, self.permutation]
        # next patch output prediction
        loss = self.compute_loss(output, target, self.block_size, mask)["loss"]
        metrics["mse_norm_pix"] = loss.item()

        return loss, metrics

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert images to patches.

        Args:
            imgs: Images of shape ``[B, 3, H, W]``.

        Returns:
            Patches of shape ``[B, (H * W) // (P ** 2), (P ** 2) * 3]``.
        """
        p = self.patch_size
        h, w = self.grid_size
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x


class MultiTokenRasterPixelLoss(RasterPixelLoss):
    """Image raster pixel loss with multi-token prediction.

    Args:
        block_size: Offsets used for the next patch prediction.
        args: Positional arguments for :class:`~l3m.loss.mae_loss.RasterPixelLoss`.
        kwargs: Keyword arguments for :class:`~l3m.loss.mae_loss.RasterPixelLoss`.
    """

    def __init__(
        self,
        block_size: list[int],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        if any(bs < 1 for bs in block_size):
            logger.warning("Block size < 1 found in the block size list.")

        self.block_size = block_size

    def forward(self, data_dict: DATA_DICT, **_: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        """Computes the loss for multiple block sizes.

        Args:
            data_dict: Input data containing the following keys:

                - ``'{self.read_key}'`` - predictions of shape ``[B, L, D]``.
                - ``'{self.target_read_key}'`` - target images or videos to be
                  :attr:`patchified <patchify>` with ``[B, 3, H, W]`` and
                  ``[B, 3, T, H, W]`` shapes, respectively.

                Optionally, can contain the following keys:

                - ``'mask'`` - mask for the patches of shape ``[B, L]``.

        Returns:
            - MSE loss
            - metrics with the following keys:

              - ``'mse_norm_pix'`` - MSE loss.
        """

        assert self.target_read_key in data_dict, data_dict.keys()

        metrics = {}
        assert isinstance(self.read_key, str)
        assert isinstance(self.target_read_key, str)
        output, target = data_dict[self.read_key], data_dict[self.target_read_key]

        mask = None
        if self.mask_read_key is not None:
            assert self.mask_read_key in data_dict
            mask = data_dict[self.mask_read_key]

        if target.ndim > 3:
            target = self.patchify(target)

        if target.shape[1] != output.shape[1]:
            target = target[:, -output.shape[1] :]

        if self.permutation is not None:
            target = target[:, self.permutation]
            output = output[:, self.permutation]

        # make different heads explicit
        B, S, _ = output.size()
        output = output.view(B, S, len(self.block_size), -1)

        # multi-token next patch prediction
        loss_dicts = [
            self.compute_loss(output[:, :, head], target, bs, mask) for head, bs in enumerate(self.block_size)
        ]

        # weight loss according to the number of targets
        total_targets = sum([d["num_targets"] for d in loss_dicts])
        loss = sum([d["loss"] * d["num_targets"] for d in loss_dicts]) / total_targets
        metrics["mse_norm_pix"] = loss.item()

        return loss, metrics


class VideoRasterPixelLoss(RasterPixelLoss):
    """Video raster pixel loss.

    Args:
        t_size: Temporal patch size.
        target_read_key: Key where target videos are stored when calling :meth:`forward`.
        kwargs: Keyword arguments for :class:`RasterPixelLoss`.
    """

    def __init__(self, t_size: int, target_read_key: str = "video", **kwargs: Any):
        super().__init__(target_read_key=target_read_key, **kwargs)
        self.t_size = t_size

    def patchify(self, video: torch.Tensor) -> torch.Tensor:
        """Convert videos to patches.

        Args:
            video: Videos of shape ``[B, 3, T, H, W]``.

        Returns:
            Patches of shape ``[B, H * W // (P * P * T), T * P * P * 3]``.
        """
        t, p = self.t_size, self.patch_size
        d = video.shape[2] // t
        h = w = video.shape[3] // p
        x = video.reshape(shape=(video.shape[0], 3, d, t, h, p, w, p))
        x = torch.einsum("ncdthpwq->ndhwtpqc", x)
        x = x.reshape(shape=(video.shape[0], d * h * w, t * p * p * 3))
        return x
