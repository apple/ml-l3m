# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import gc
import json
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf.dictconfig import DictConfig
from torch.optim import Optimizer

from l3m.engine import evaluate
from l3m.helpers.checkpoint import DistributedCheckpointUtils, save_on_master
from l3m.helpers.dist import utils as dist_utils
from l3m.optim import scaler as optim_scaler

__all__ = [
    "maybe_eval_and_save_ckpt",
    "aggregate_metrics",
]

logger = logging.getLogger("l3m")

AGGREGATE_METRICS = ["CIDEr", "Accuracy"]


def maybe_eval_and_save_ckpt(
    cfg: DictConfig,
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: Callable,
    loss_scaler: optim_scaler.NativeScaler,
    data_loader_train: Iterable,
    data_loaders_val: Iterable,
    device: torch.device,
    n_parameters: int,
    total_iters: int,
    metrics_computer: Callable,
    iteration: int | None = None,
    criterion: Callable | None = None,
    tokenizer: Callable | None = None,
    dtype: torch.dtype | None = None,
    **_: Any,
):
    """Performs evaluation and saves model checkpoints based on specified frequencies.

    Args:
        cfg: Hydra configuration object.
        model: Model to evaluate and save.
        optimizer: Optimizer used for training.
        lr_scheduler: Learning rate scheduler.
        loss_scaler: Gradient scaler.
        data_loader_train: Train data loader.
        data_loaders_val: Validation data loaders.
        device: Torch device.
        n_parameters: Number of model parameters.
        total_iters: Total training iterations.
        metrics_computer: Function to compute evaluation metrics.
        iteration: Current training iteration.
        criterion: Loss function.
        tokenizer: Tokenizer.
        dtype: Data type.
    """

    gc.collect()
    torch.cuda.empty_cache()

    exp_cfg, data_cfg = cfg["experiment"], cfg["data"]
    output_dir = Path(exp_cfg["output_dir"])

    ckpt_save_freq = exp_cfg["ckpt_save_freq"] or float("inf")
    test_frequency = exp_cfg["test_frequency"] or float("inf")
    save_full_checkpoint_on_main = exp_cfg.get("save_full_checkpoint_on_main", True)

    if exp_cfg["output_dir"]:
        if (iteration + 1) % ckpt_save_freq == 0 or (iteration == total_iters):
            checkpoint_paths = [
                output_dir / f"checkpoint_{iteration}",
                Path(exp_cfg["shared_dir"]) / "checkpoint",
            ]
            for checkpoint_path in checkpoint_paths:
                DistributedCheckpointUtils.save(
                    checkpoint_path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    extra={
                        "cfg": cfg,
                        "iteration": iteration,
                        "lr_scheduler": (
                            lr_scheduler.state_dict() if hasattr(lr_scheduler, "state_dict") else lr_scheduler
                        ),
                        "loss_scaler": loss_scaler.state_dict(),
                        "data_loader_train": (
                            data_loader_train.state_dict() if hasattr(data_loader_train, "state_dict") else None
                        ),
                    },
                )
                # save the full checkpoint in the main node
                if save_full_checkpoint_on_main:
                    state_dict = DistributedCheckpointUtils.create_state_dict(
                        model=model,
                        optimizer=optimizer,
                        extra={
                            "cfg": cfg,
                            "iteration": iteration,
                            "lr_scheduler": (
                                lr_scheduler.state_dict() if hasattr(lr_scheduler, "state_dict") else lr_scheduler
                            ),
                            "loss_scaler": loss_scaler.state_dict(),
                            "data_loader_train": (
                                data_loader_train.state_dict() if hasattr(data_loader_train, "state_dict") else None
                            ),
                        },
                        gather_dtensors=save_full_checkpoint_on_main,
                    )
                    save_on_master(state_dict, checkpoint_path.with_suffix(".pth"))

    if data_cfg["validation"] and (((iteration + 1) % test_frequency == 0) or (iteration == total_iters)):
        checkpoint_paths = [
            output_dir / "checkpoint",
            Path(exp_cfg["shared_dir"]) / "checkpoint",
        ]
        for checkpoint_path in checkpoint_paths:
            DistributedCheckpointUtils.save(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                extra={
                    "cfg": cfg,
                    "iteration": iteration,
                    "lr_scheduler": (
                        lr_scheduler.state_dict() if hasattr(lr_scheduler, "state_dict") else lr_scheduler
                    ),
                    "loss_scaler": loss_scaler.state_dict(),
                    "data_loader_train": (
                        data_loader_train.state_dict() if hasattr(data_loader_train, "state_dict") else None
                    ),
                },
            )
            # save the full checkpoint in the main node
            if save_full_checkpoint_on_main:
                state_dict = DistributedCheckpointUtils.create_state_dict(
                    model=model,
                    optimizer=optimizer,
                    extra={
                        "cfg": cfg,
                        "iteration": iteration,
                        "lr_scheduler": (
                            lr_scheduler.state_dict() if hasattr(lr_scheduler, "state_dict") else lr_scheduler
                        ),
                        "loss_scaler": loss_scaler.state_dict(),
                        "data_loader_train": (
                            data_loader_train.state_dict() if hasattr(data_loader_train, "state_dict") else None
                        ),
                    },
                    gather_dtensors=save_full_checkpoint_on_main,
                )
                save_on_master(state_dict, checkpoint_path.with_suffix(".pth"))

        test_stats = {}
        for dataset_name, data_loader_val in data_loaders_val.items():
            test_stat = evaluate(
                data_loader=data_loader_val,
                model=model,
                device=device,
                iteration=iteration,
                criterion=criterion,
                metrics_computer=metrics_computer,
                tokenizer=tokenizer,
                dtype=dtype,
                dataset_name=dataset_name,
            )
            test_stats.update(test_stat)

        test_stats = aggregate_metrics(test_stats)
        log_stats = {
            **{f"test_{k}": v for k, v in test_stats.items()},
            "iteration": iteration,
            "n_parameters": n_parameters,
        }

        if exp_cfg["output_dir"] and dist_utils.is_main_process():
            with (output_dir / "legacy_logs.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    gc.collect()
    torch.cuda.empty_cache()


def aggregate_metrics(stats: dict[str, Any]) -> dict[str, Any]:
    """Averages specified metrics across all processes in a distributed setting.

    Args:
        stats: A dictionary containing the metrics to aggregate.

    Returns:
        Dictionary containing the aggregated metrics, and the original metrics.
    """

    averaged_metrics = {}
    for k, v in stats.items():
        for m in AGGREGATE_METRICS:
            if m in k:
                if m in averaged_metrics:
                    averaged_metrics[m].append(v)
                else:
                    averaged_metrics[m] = [v]
                break
    averaged_metrics = {k: sum(v) / len(v) for k, v in averaged_metrics.items()}
    avg_score = 0
    avg_score_name = ""
    if len(averaged_metrics) > 1:
        for k, v in averaged_metrics.items():
            avg_score += v
            avg_score_name += f"{k}_"
        if avg_score_name:
            averaged_metrics.update({avg_score_name: avg_score})

    stats.update(averaged_metrics)
    return stats
