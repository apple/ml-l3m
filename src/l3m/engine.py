# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import logging
import math
import sys
from collections.abc import Callable, Iterable
from typing import Any

import torch
import wandb
from torch.optim import Optimizer

from l3m.helpers.dist import utils as dist_utils
from l3m.helpers.dist.cp import create_context_parallel_ctx, get_train_context
from l3m.helpers.dist.utils import DeviceMeshHandler
from l3m.helpers.utils import chunk_batch, move_to_device
from l3m.logging import DatasetLogger, MetricLogger, SmoothedValue
from l3m.optim import scaler as optim_scaler
from l3m.optim.scheduler import step_scheduler

__all__ = ["train", "evaluate"]

logger = logging.getLogger("l3m")


def train(
    model: torch.nn.Module,
    criterion: Callable,
    data_loader: Iterable,
    optimizer: Optimizer,
    lr_scheduler: Callable,
    device: torch.device,
    total_iters: int,
    evaluator_callback: Callable,
    loss_scaler: optim_scaler.NativeScaler,
    fsdp_extras: dict[Any, Any],
    max_norm: float | None = 0,
    set_training_mode: bool = True,
    dtype: torch.dtype = torch.float16,
    amp_enabled: bool = True,
    test_frequency: int | None = None,
    ckpt_save_freq: int | None = None,
    start_iteration: int = 0,
    gradient_accumulation_steps: int = 1,
    enable_loss_parallel: bool = False,
) -> dict[str, Any]:
    """Trains a model with all the bells and whistles, e.g., gradient accumulation,
    moving data to the correct device, logging.

    Args:
        model: model to train.
        criterion: loss function that receives the data_dict
            which contains both input data and the outputs of the model.
        data_loader: train dataset already initialized and properly sharded.
        optimizer: pre-initialized optimizer.
        lr_scheduler: learning rate scheduler.
        device: torch device of the current (shard of the) model.
        total_iters: number of batches to train for.
        evaluator_callback: pre-built evaluator.
        loss_scaler: gradient scaler that also does gradient clipping.
        fsdp_extras: Extra configurations for the FSDP setup, such as the context_parallel_plan.
        max_norm: max norm for gradient clipping.
        set_training_mode: whether to set the model to train.
        dtype: dtype for autocast.
        amp_enabled: whether to perform automatic mixed precision.
        test_frequency: number of iterations to do before performing an evaluation
        ckpt_save_freq: number of iterations to do before saving the model.
        start_iteration: start iteration, used to skip ahead if resuming training.
        gradient_accumulation_steps: number of batches to perform gradient accumulation.
            In practice, the batch size will be divided by this value. This means that you should set your
            desired per-gpu batch size and then play around with this parameter until you can fit everything in the GPU.
        enable_loss_parallel: Whether to enable loss parallel for input sharded along the
            class dimension.

    Returns:
        A dictionary containing the computed metrics.
    """

    model.train(set_training_mode)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    data_logger = DatasetLogger()
    print_freq = 10

    train_context = get_train_context(enable_loss_parallel)

    for it, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, start=start_iteration, max_iterations=total_iters)
    ):
        iteration = it + start_iteration
        step_scheduler(
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
            where=float(iteration) / total_iters,
        )

        batch = move_to_device(batch, device=device)

        for chunked_batch in chunk_batch(batch, gradient_accumulation_steps):
            # apply context parallelism if cp is enabled
            if DeviceMeshHandler.cp_enabled():
                optional_context_parallel_ctx = create_context_parallel_ctx(
                    cp_mesh=DeviceMeshHandler.device_mesh["cp"],
                    cp_plan=fsdp_extras.get("context_parallel_plan", None),
                    data_dict=chunked_batch,
                    # For Pipeline Parallel the model may be made up of multiple model parts.
                    model_parts=[model],
                )
            else:
                optional_context_parallel_ctx = None

            with train_context(optional_context_parallel_ctx):
                with torch.autocast("cuda", enabled=amp_enabled, dtype=dtype):
                    output = model(chunked_batch)

                    loss, metrics = criterion(output, model=model)
                    # Free before backward to avoid peaking memory.
                    del output
                    loss_value = loss.item()

                    if not math.isfinite(loss_value):
                        logger.info(f"Loss is {loss_value}, stopping training")
                        sys.exit(1)

                    loss = loss / gradient_accumulation_steps
                loss_scaler.backward(loss)

        loss_scaler(optimizer, clip_grad=max_norm, model=model)

        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()

        if dist_utils.is_main_process() and wandb.run is not None:
            wandb.log(
                data={
                    "iteration": iteration,
                    "train_loss": loss_value,
                    "lr": optimizer.param_groups[0]["lr"],
                    **metrics,
                },
                commit=True,
            )

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(**metrics)
        data_logger.update(batch=batch)

        do_test_callback = test_frequency is not None and (iteration + 1) % test_frequency == 0
        do_ckpt_save_callback = ckpt_save_freq is not None and (iteration + 1) % ckpt_save_freq == 0
        if do_test_callback or do_ckpt_save_callback:
            evaluator_callback(iteration=iteration)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats:{str(metric_logger)}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    iteration: int,
    criterion: Callable,
    metrics_computer: Callable,
    dataset_name: str = "",
    **_: Any,
) -> dict[str, Any]:
    """Evaluates a model on the given dataset.

    Args:
        model: model to evaluate.
        data_loader: validation dataset already initialized and properly sharded.
        device: torch device of the current (shard of the) model.
        iteration: current training iteration, used for logging.
        criterion: loss function that receives the data_dict
            which contains both input data and the outputs of the model.
        metrics_computer: function to compute all the performance metrics.
        dataset_name: Name of the dataset being evaluated.

    Returns:
        A dictionary containing the computed metrics.
    """

    metric_logger = MetricLogger(delimiter="  ")
    header = f"Test {dataset_name}:"
    prefix = f"test_{dataset_name}"

    model.eval()  # switch to evaluation mode

    for batch_id, batch in enumerate(metric_logger.log_every(data_loader, 10, header, max_iterations=len(data_loader))):
        batch = move_to_device(batch, device=device)
        batch_size = next(iter(batch.values())).shape[0]
        with torch.amp.autocast("cuda", enabled=False):
            output = model(batch)
            loss, stats = criterion(output, model=model)

        output["dataset_name"] = dataset_name
        stats.update(
            metrics_computer(
                data_dict=output,
                model=model,
                iteration=iteration,
                reset=(batch_id == 0),
            )
        )

        for k, v in stats.items():
            metric_logger.meters[f"{prefix}_{k}"].update(v, n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats.update(metrics_computer.apply_postprocessing(stats, prefix=prefix))
    logger.info(str.join(", ", [f"{k}: {v}" for k, v in stats.items()]))

    if dist_utils.is_main_process() and wandb.run is not None:
        wandb.log(
            data={
                "iteration": iteration,
                **stats,
            },
            commit=True,
        )

    model.train()  # switch back to training mode

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
