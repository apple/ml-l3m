# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import logging
import os
import shutil
from collections.abc import Callable, Iterable
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import omegaconf
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.distributed.tensor
import torch.nn as nn
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.optim import Optimizer

from l3m.constants.typing import CHECKPOINT
from l3m.helpers.dist import utils as dist_utils
from l3m.optim import scaler as optim_scaler

__all__ = [
    "prepare_and_load_checkpoints",
    "load_checkpoint",
    "filter_checkpoint",
    "save_on_master",
    "rename_checkpoint",
    "dist_execute_only_main",
    "DistributedCheckpointUtils",
]

logger = logging.getLogger("l3m")


def prepare_and_load_checkpoints(model_cfg: DictConfig) -> CHECKPOINT:
    """Loads and prepares checkpoints for a model.

    It supports loading multiple checkpoints, filtering and renaming keys,
    resizing position embeddings, and resizing to new vocabulary sizes.

    Args:
        model_cfg: Model configuration containing checkpoint information.

    Returns:
        A dictionary containing the combined checkpoint data.
    """

    # dict to aggregate all checkpoints
    # checkpoints with later position in the list will overwrite
    # the previous keys if overlapping
    combined_checkpoint = {}

    if not isinstance(model_cfg["checkpoint"], ListConfig):
        # This covers the case where the checkpoints and its related operations (e.g. filter, renamer, ..)
        # are placed in a flat manner in the model config.
        ckpt_loading_cfg = [model_cfg]
    else:
        ckpt_loading_cfg = model_cfg["checkpoint"]

    for ckpt in ckpt_loading_cfg:
        checkpoint = load_checkpoint(ckpt["checkpoint"])
        checkpoint_model = checkpoint["model"] if "model" in checkpoint else checkpoint
        checkpoint_model = filter_checkpoint(ckpt.get("checkpoint_filters", None), checkpoint_model)
        checkpoint_model = rename_checkpoint(ckpt.get("checkpoint_renamer", None), checkpoint_model)
        combined_checkpoint.update(checkpoint_model)

    return combined_checkpoint


def load_checkpoint(ckpt_path: str) -> CHECKPOINT:
    """Loads a checkpoint from the specified path.

    Args:
        ckpt_path: Path to the checkpoint file.

    Returns:
        A dictionary containing the checkpoint data.
    """

    logger.info(f"Loading checkpoint from given path: {ckpt_path}")
    return torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)


def filter_checkpoint(
    filters: Iterable[str] | None,
    checkpoint: CHECKPOINT,
) -> CHECKPOINT:
    """Filters keys from a checkpoint based on a list of glob-style filters.

    Args:
        filters: A list of glob-style filters.
        checkpoint: The checkpoint dictionary.

    Returns:
        The filtered checkpoint dictionary.
    """

    if filters is None:
        return checkpoint

    delete_keys = []
    for k in checkpoint:
        if any(fnmatch(k, excl) for excl in filters):
            delete_keys.append(k)

    for k in delete_keys:
        del checkpoint[k]

    return checkpoint


def rename_checkpoint(mappings: list[tuple[str, ...]] | None, checkpoint: CHECKPOINT) -> CHECKPOINT:
    """Renames keys in a checkpoint based on a list of mappings.

    Args:
        mappings: A list of (source, target) rename mappings.
        checkpoint: The checkpoint dictionary.

    Returns:
        The checkpoint dictionary with renamed keys.
    """

    if mappings is None:
        return checkpoint

    for src, target in mappings:
        checkpoint = {x.replace(src, target): y for x, y in checkpoint.items()}

    return checkpoint


def save_on_master(*args, **kwargs) -> None:
    """Saves only on master."""
    torch.distributed.barrier()  # Ensures all processes are synchronized
    if dist_utils.is_main_process():
        torch.save(*args, **kwargs)


def dist_execute_only_main(fn: Callable[[Any], Any], *args: Any, **kwargs: Any) -> None:
    """Executes function only on the main process."""
    if dist.is_initialized():
        dist.barrier()
        if dist_utils.is_main_process():
            fn(*args, **kwargs)
        dist.barrier()
    else:
        fn(*args, **kwargs)


class DistributedCheckpointUtils:
    """Utility class that provides tools for creating, saving and loading FSDP2 checkpoints with DCP.

    Note 1: DCP creates a folder instead of saving in a file like torch.save().

    Note 2: DCP does NOT store anything apart from the model and optimizer state_dicts().
    Because of this, we need to manually store everything else that we care in an extra file.
    This class handles all of that automatically, so you don't need to think about it.
    """

    MODEL_KEY = "model"
    OPTIMIZER_KEY = "optimizer"
    OTHER_PATH = "other.pth"

    @staticmethod
    def create_state_dict(
        model: nn.Module,
        optimizer: Optimizer,
        extra: Any,
        gather_dtensors: bool = False,
    ) -> dict[str, Any]:
        model_sd, optim_sd = get_state_dict(model, optimizer)
        # gathers dtensors across devices by calling full_tensor()
        if gather_dtensors:
            model_sd = {
                k: (v.full_tensor() if isinstance(v, torch.distributed.tensor.DTensor) else v)
                for k, v in model_sd.items()
            }
            optim_sd["state"] = {
                k: (v.full_tensor() if isinstance(v, torch.distributed.tensor.DTensor) else v)
                for k, v in optim_sd["state"].items()
            }

        state_dict = {
            DistributedCheckpointUtils.MODEL_KEY: model_sd,
            DistributedCheckpointUtils.OPTIMIZER_KEY: optim_sd,
            **extra,
        }
        return state_dict

    @staticmethod
    def save(
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optimizer,
        extra: Any,
        remove_if_exists: bool = True,
    ):
        if dist.is_initialized():
            dist.barrier()

        state_dict = DistributedCheckpointUtils.create_state_dict(
            model=model,
            optimizer=optimizer,
            extra=extra,
            gather_dtensors=False,
        )

        if remove_if_exists and os.path.exists(checkpoint_path) and dist_utils.is_main_process():
            shutil.rmtree(checkpoint_path)

        dcp_state_dict = {
            DistributedCheckpointUtils.MODEL_KEY: state_dict[DistributedCheckpointUtils.MODEL_KEY],
            DistributedCheckpointUtils.OPTIMIZER_KEY: state_dict[DistributedCheckpointUtils.OPTIMIZER_KEY],
        }

        dcp.save(dcp_state_dict, checkpoint_id=checkpoint_path)

        # dcp doesn't save anything apart from model/optimizer state_dict
        not_saved_by_dcp = {}
        for k, v in state_dict.items():
            if k not in [
                DistributedCheckpointUtils.MODEL_KEY,
                DistributedCheckpointUtils.OPTIMIZER_KEY,
            ]:
                not_saved_by_dcp[k] = v

        torch.save(
            not_saved_by_dcp,
            os.path.join(checkpoint_path, DistributedCheckpointUtils.OTHER_PATH),
        )

        if dist.is_initialized():
            dist.barrier()

    @staticmethod
    def load(state_dict: dict[str, Any], checkpoint_path: str) -> dict[str, Any]:
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if k
            in [
                DistributedCheckpointUtils.MODEL_KEY,
                DistributedCheckpointUtils.OPTIMIZER_KEY,
            ]
        }
        dcp.load(state_dict, checkpoint_id=checkpoint_path)

        # manually load stuff that dcp can't save :)
        other_stuff = os.path.join(checkpoint_path, DistributedCheckpointUtils.OTHER_PATH)
        if os.path.exists(other_stuff):
            for k, v in torch.load(other_stuff, weights_only=False).items():
                state_dict[k] = v

        return state_dict

    @staticmethod
    def dcp_checkpoint_to_torch(ckpt_dir: str, consolidate_path: str) -> str:
        """Consolidates all FSDP checkpoints in a directory to a single file.

        Consolidate checkpoint is saved in a subdirectory of ckpt_dir

        Adapted from https://github.com/facebookresearch/lingua/

        Args:
            ckpt_dir: path to the directory containing the checkpoints.
            consolidate_path: path where to save the consolidated checkpoint.

        Returns:
            The path to the consolidated checkpoint.
        """

        consolidate_path = Path(consolidate_path)
        consolidate_path.parent.mkdir(exist_ok=True)
        logger.info(f"Consolidating to: {str(consolidate_path)}")

        # save dcp to torch
        dcp_to_torch_save(ckpt_dir, str(consolidate_path))

        # load that same checkpoint to add other keys
        other_stuff = os.path.join(ckpt_dir, DistributedCheckpointUtils.OTHER_PATH)
        if os.path.exists(other_stuff):
            checkpoint = torch.load(
                str(consolidate_path),
                map_location="cpu",
                weights_only=False,
            )
            for k, v in torch.load(other_stuff, weights_only=False).items():
                checkpoint[k] = v
            torch.save(checkpoint, str(consolidate_path))

        logger.info("Consolidated!")
        return str(consolidate_path)

    @staticmethod
    def resume_training(
        cfg: omegaconf.DictConfig,
        exp_cfg: omegaconf.DictConfig,
        local_rank: int,
        model: nn.Module,
        optimizer: Optimizer,
        data_loader_train: Iterable,
        lr_scheduler: Callable,
        loss_scaler: optim_scaler.NativeScaler,
    ) -> None:
        del local_rank
        ckpt_path = exp_cfg["resume"]
        checkpoint = DistributedCheckpointUtils.create_state_dict(
            model=model,
            optimizer=optimizer,
            extra={
                "cfg": cfg,
                "iteration": 0,
                "lr_scheduler": (lr_scheduler.state_dict() if hasattr(lr_scheduler, "state_dict") else lr_scheduler),
                "loss_scaler": loss_scaler.state_dict(),
                "data_loader_train": (
                    data_loader_train.state_dict() if hasattr(data_loader_train, "state_dict") else None
                ),
            },
        )
        checkpoint = DistributedCheckpointUtils.load(checkpoint, ckpt_path)
        set_state_dict(
            model,
            optimizer,
            model_state_dict=checkpoint["model"],
            optim_state_dict=checkpoint["optimizer"],
        )

        # iteration
        if not exp_cfg["ignore_resume_iteration"]:
            exp_cfg["start_iteration"] = checkpoint["iteration"] + 1

        # lr scheduler
        if hasattr(lr_scheduler, "load_state_dict"):
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        # scaler
        if "scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["scaler"])

        # dataloader
        if hasattr(data_loader_train, "load_state_dict"):
            data_loader_train.load_state_dict(checkpoint["data_loader_train"])
