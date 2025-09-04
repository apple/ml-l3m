# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import argparse
import logging
import os
from pathlib import Path

import omegaconf
import runner
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import OmegaConf

from l3m.helpers import omegaconf_resolvers  # noqa
from l3m.helpers.dist import utils as dist_utils
from l3m.logging import setup_loggers

DATA_DIR = "/mnt/data/"
DEFAULT_CONFIG = "configs/defaults.yaml"


def build_cfg(args: argparse.Namespace, cli_cfg: list[str]) -> tuple[str, omegaconf.DictConfig]:
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    name = Path(args.config).stem
    parameters = config

    # 1. default config
    cfg = OmegaConf.to_container(OmegaConf.load(DEFAULT_CONFIG), resolve=True)

    # 2. overwrite using the target config
    cfg = OmegaConf.merge(cfg, parameters)

    # 3. overwrite using command line options
    cli_cfg = OmegaConf.from_cli(cli_cfg)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    assert isinstance(cfg, omegaconf.DictConfig)

    return name, cfg


def main(args: argparse.Namespace, cli_cfg: list[str]) -> None:
    name, cfg = build_cfg(args, cli_cfg)

    cfg["experiment"]["output_dir"] = "/tmp"
    cfg["experiment"]["shared_dir"] = "/tmp"

    ext = "" if cfg["experiment"]["fsdp"] is not None else ".pth"
    checkpoint_file = os.path.join(cfg["experiment"]["shared_dir"], f"checkpoint{ext}")
    if os.path.exists(checkpoint_file):
        cfg["experiment"]["resume"] = checkpoint_file

    # init distributed environment
    dist_utils.init_distributed_mode(cfg)

    try:  # Download torchvision datasets in the launch script
        if (
            "torchvision" in cfg.data.train.dataset._target_
            or "trainval" in cfg.data.train.dataset._target_
            or "tfds" in cfg.data.train.dataset._target_
        ) and dist_utils.is_main_process():  # dist_utils.get_local_rank() == 0:
            instantiate(cfg.data.train.dataset, download=True)  # Force download
    except Exception as e:  # noqa: BLE001
        logging.getLogger("l3m").warning(f"Torchvision dataset loading error: {e}.")

    # Wait till mounting or downloads finish.
    if dist.is_initialized():
        dist.barrier()

    setup_loggers(name=name, args=args, cfg=cfg)

    # Launch
    runner.main(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--debug", action="store_true")

    args, cli_cfg = parser.parse_known_args()

    main(args, cli_cfg)
