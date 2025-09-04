# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""Adapted from:

- https://github.com/facebookresearch/dinov2/blob/main/dinov2/logging/__init__.py
"""

import argparse
import datetime
import functools
import logging
import os
import sys
import time
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from typing import Any

import omegaconf
import torch
import torch.distributed as dist
import wandb
import yaml
from omegaconf import OmegaConf

from l3m.constants.generic import DATASET_NAME_KEY
from l3m.helpers.dist import utils as dist_utils

__all__ = [
    "setup_loggers",
    "setup_logging",
    "setup_wandb",
    "SmoothedValue",
    "MetricLogger",
    "DatasetLogger",
]

logger = logging.getLogger("l3m")


# So that calling _configure_logger multiple times won't add many handlers
@functools.lru_cache
def _configure_logger(
    name: str = None,
    level: int = logging.DEBUG,
    output: str = None,
):
    """Configures a logger with specified settings.

    Args:
        name: Logger name. Defaults to None.
        level: Logging level. Defaults to logging.DEBUG.
        output: Output file path. Defaults to None, which means output to standard out.

    Returns:
        The configured logger.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Loosely match Google glog format:
    #   [IWEF]yyyymmdd hh:mm:ss.uuuuuu threadid file:line] msg
    # but use a shorter timestamp and include the logger name:
    #   [IWEF]yyyymmdd hh:mm:ss logger threadid file:line] msg
    fmt = "%(levelname).1s%(asctime)s %(process)s %(name)s %(filename)s:%(lineno)s] %(message)s"
    datefmt = "%Y%m%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # stdout logging for main worker only
    if dist_utils.is_main_process():
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # file logging for all workers
    if output:
        if os.path.splitext(output)[-1] in (".txt", ".log"):
            filename = output
        else:
            filename = os.path.join(output, "logs", "log.txt")

        if not dist_utils.is_main_process():
            global_rank = dist_utils.DeviceMeshHandler.get_rank_info().global_rank
            filename = filename + f".rank{global_rank}"

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        handler = logging.StreamHandler(open(filename, "a"))
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def setup_logging(
    output: str = None,
    name: str = None,
    level: int = logging.DEBUG,
    capture_warnings: bool = True,
):
    """Sets up logging.

    Args:
        output: Path to the log file. If None, logs to stdout.
        name: Name of the logger.
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
        capture_warnings: Whether to capture warnings and redirect them to the logging system.
    """

    logging.captureWarnings(capture_warnings)
    _configure_logger(name, level=level, output=output)


def setup_wandb(name: str, wandb_cfg: dict[str, str], cfg: omegaconf.DictConfig, **wandb_kwargs: Any) -> None:
    """Initializes a wandb run using the config.

    Args:
        name: name of the wandb run.
        wandb_cfg: dictionary containing wandb-specific configurations.
        cfg: omegaconf configuration.
        wandb_kwargs: extra wandb kwargs.
    """

    with open(wandb_cfg["wandb_config"]) as fin:
        wandb_config = yaml.safe_load(fin)

    os.environ["WANDB_BASE_URL"] = wandb_config["host-name"]
    os.environ["WANDB_API_KEY"] = wandb_config["api-key"]
    wandb.init(
        name=name,
        entity=wandb_config["entity"],
        project=wandb_cfg["project"],
        tags=wandb_cfg["tags"],
        config=OmegaConf.to_container(cfg),
        **wandb_kwargs,
    )


def setup_loggers(name: str, args: argparse.Namespace, cfg: omegaconf.DictConfig) -> None:
    """Sets up logging and wandb.

    Args:
        name: Experiment name.
        args: Command line arguments.
        cfg: Configuration object.
    """

    setup_logging(
        name="l3m",
        output=os.path.join(cfg["experiment"]["output_dir"], "log.txt"),
    )

    if cfg["wandb"]["use_wandb"] and dist_utils.is_main_process() and not args.debug:
        try:
            setup_wandb(name, cfg["wandb"], cfg)
        except Exception as e:  # noqa: BLE001
            logging.getLogger("l3m").warning(f"Wandb init failed with error: {e}.\nRunning without wandb.")


class SmoothedValue:
    """Tracks a series of values and provides smoothed statistics.

    It maintains a deque of recent values and calculates median, average,
    global average, and maximum values. It also synchronizes the total and
    count across distributed processes.

    Args:
        window_size: Size of the moving average window.
        fmt: Format string for printing the smoothed value.
    """

    def __init__(
        self,
        window_size: int = 20,
        fmt: str = "{median:.4f} ({global_avg:.4f})",
    ):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1) -> None:
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self) -> None:
        """Warning: does not synchronize the deque!"""
        if dist_utils.is_dist_avail_and_initialized():
            t = torch.tensor(
                [self.count, self.total],
                dtype=torch.float64,
                device="cuda",
            )
            dist.barrier()
            dist.all_reduce(t)
            t = t.tolist()
            self.count = int(t[0])
            self.total = t[1]

    @property
    def median(self) -> float:
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self) -> float:
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self) -> float:
        return self.total / self.count

    @property
    def max(self) -> float:
        return max(self.deque)

    @property
    def value(self) -> float:
        return self.deque[-1]

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    """A class to track and log metrics during training.

    It stores metrics using `SmoothedValue` for smoothing, and provides
    functionality to update, synchronize, and log these metrics.

    Args:
        delimiter: string delimiter for logging.
    """

    def __init__(self, delimiter: str = "  "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            self.meters[k].update(v)

    def __getattr__(self, attr: str) -> SmoothedValue:
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self) -> None:
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name: str, meter: SmoothedValue) -> None:
        self.meters[name] = meter

    def log_every(
        self,
        iterable: Iterable,
        print_freq: int,
        header: str = "",
        start: int = 0,
        max_iterations: int = None,
    ):
        i = start
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")

        if max_iterations is None:
            max_iterations = len(iterable)

        space_fmt = ":" + str(len(str(max_iterations))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == max_iterations - 1:
                eta_seconds = iter_time.global_avg * (max_iterations - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(
                        log_msg.format(
                            i,
                            max_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    logger.info(
                        log_msg.format(
                            i,
                            max_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
            if dist_utils.is_main_process() and wandb.run is not None:
                wandb.log(
                    data={
                        "Time/ETA(mins)": eta_seconds // 60,
                        "Time/iter": iter_time.value,
                        "Time/data": data_time.value,
                    },
                    commit=False,
                )

            if i >= max_iterations:
                break

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"{header} Total time: {total_time_str} ({total_time / max_iterations:.4f} s / it)")


class DatasetLogger:
    """Logs the number of samples seen from each dataset."""

    def __init__(self):
        self.dataset_to_count_dict = defaultdict(int)

    def update(self, batch: dict[str, Any]) -> None:
        if DATASET_NAME_KEY not in batch:
            return

        if dist_utils.is_main_process() and wandb.run is not None:
            histogram = Counter(batch[DATASET_NAME_KEY])
            for k, v in histogram.items():
                self.dataset_to_count_dict[k] += v

            samples_seen = {f"samples_seen/{k}": v for k, v in self.dataset_to_count_dict.items()}
            wandb.log(
                data={
                    "samples_seen/total": sum(self.dataset_to_count_dict.values()),
                    **samples_seen,
                },
                commit=False,
            )
