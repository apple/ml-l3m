# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import datetime
import logging
import time
from collections.abc import Iterable
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler

from l3m.data.samplers import InfiniteDataSampler
from l3m.engine import evaluate, train
from l3m.evaluator import maybe_eval_and_save_ckpt
from l3m.helpers import checkpoint as ckpt_helpers
from l3m.helpers.checkpoint import DistributedCheckpointUtils
from l3m.helpers.dist import fsdp
from l3m.helpers.dist import utils as dist_utils
from l3m.helpers.dist.utils import DeviceMeshHandler
from l3m.helpers.utils import init_parameters
from l3m.optim import scaler as optim_scaler
from l3m.optim.optimizer import create_optimizer

logger = logging.getLogger("l3m")


def main(cfg) -> None:
    logger.info(OmegaConf.to_yaml(cfg))

    exp_cfg, data_cfg, model_cfg = cfg["experiment"], cfg["data"], cfg["model"]
    device = torch.device(exp_cfg["device"])

    # fix the seed for reproducibility
    device_mesh = DeviceMeshHandler.get_device_mesh(exp_cfg)
    rank_info = DeviceMeshHandler.get_rank_info()
    logger.info(rank_info)

    model_rank, world_size = rank_info.model_rank, rank_info.world_size
    seed = exp_cfg["seed"] + model_rank

    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002

    cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = 1000
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dataset_train = instantiate(data_cfg["train"]["dataset"])
    if data_cfg["validation"]:
        datasets_val = instantiate(data_cfg["validation"]["dataset"])
        if not isinstance(datasets_val, DictConfig):
            datasets_val = {"dataset": datasets_val}

    if exp_cfg["distributed"]:
        sampler_train = None
        if dataset_train is not None:
            sampler_train = InfiniteDataSampler(
                dataset=dataset_train,
                num_replicas=world_size,
                rank=model_rank,
                shuffle=data_cfg["train"]["shuffle"],
            )

        if data_cfg["validation"]:
            samplers_val = {}
            for dataset_name, dataset_val in datasets_val.items():
                # Only create Distributed Samplers if Dataloader is not None,
                # Otherwise, the dataset object is expected to return an iterator
                # and handle the DataParallel requirements (e.g. tfrecord dataloader)
                if data_cfg["validation"]["dataloader"] is not None:
                    if cfg["experiment"]["dist_eval"]:
                        if len(dataset_val) % world_size != 0:
                            logger.info(
                                "Warning: Enabling distributed evaluation with an eval dataset not divisible by "
                                "the number of processes. This will slightly alter validation results as extra "
                                "duplicate entries are added to achieve equal num of samples per-process."
                            )
                        sampler_val = DistributedSampler(
                            dataset_val,
                            num_replicas=world_size,
                            rank=model_rank,
                            shuffle=False,
                        )
                    else:
                        sampler_val = SequentialSampler(dataset_val)
                    samplers_val[dataset_name] = sampler_val
    else:
        sampler_train = None
        if dataset_train is not None:
            sampler_train = RandomSampler(dataset_train)
        if data_cfg["validation"]:
            samplers_val = {}
            for dataset_name, dataset_val in datasets_val.items():
                sampler_val = SequentialSampler(dataset_val)
                samplers_val[dataset_name] = sampler_val

    validation_tokenizer = instantiate(data_cfg["validation_tokenizer"]) or instantiate(data_cfg["tokenizer"])

    collate_fn = instantiate(data_cfg["train"]["collator"])
    data_loader_train = instantiate(data_cfg["train"]["dataloader"])(
        dataset=dataset_train,
        sampler=sampler_train,
        collate_fn=collate_fn,
    )

    data_loaders_val = {}
    if data_cfg["validation"]:
        val_collate_fn = instantiate(data_cfg["validation"].get("collator", None))
        if data_cfg["validation"]["dataloader"] is not None:
            for dataset_name, dataset_val in datasets_val.items():
                data_loader_val_fn = instantiate(data_cfg["validation"]["dataloader"])
                if val_collate_fn:
                    data_loader_val = data_loader_val_fn(
                        dataset=dataset_val,
                        sampler=samplers_val[dataset_name],
                        collate_fn=val_collate_fn,
                    )
                else:
                    data_loader_val = data_loader_val_fn(dataset=dataset_val, sampler=samplers_val[dataset_name])

                data_loaders_val[dataset_name] = data_loader_val
                if hasattr(dataset_val, "len"):
                    logger.info(f"dataset_name: {dataset_name}, size: {len(dataset_val)}")
        else:
            # If dataloader is None, the dataset itself is expected
            # to return an iterator and handle the DataParallel
            # requirements (e.g. tfrecord dataloader)
            assert len(datasets_val) > 0
            assert isinstance(iter(datasets_val.values()).__next__(), Iterable), (
                "The dataset object needs to be an `Iterable` if dataloader is None."
            )
            data_loaders_val = datasets_val

    use_meta_device = not model_cfg["checkpoint"]
    if use_meta_device:
        logger.info("Not loading from checkpoint, using meta device to initialize")
        with torch.device("meta"):
            model = instantiate(model_cfg["meta_model"])
        logger.info(f"Created model:\n{model}")

    else:
        model = instantiate(model_cfg["meta_model"])
        logger.info(f"Created model:\n{model}")
        if model_cfg["checkpoint"] and dist_utils.get_local_rank() <= 0:
            checkpoint_model = ckpt_helpers.prepare_and_load_checkpoints(model_cfg=model_cfg)
            status = model.load_state_dict(checkpoint_model, strict=False)
            logger.info(str(status))

        model.to(device)

    if exp_cfg["distributed"]:
        if exp_cfg["fsdp"] is not None:
            model, fsdp_extras = fsdp.init_fsdp2(
                model=model,
                device_mesh=device_mesh,
                exp_cfg=exp_cfg,
                broadcast=not use_meta_device,
            )
        else:
            raise ValueError("DDP is not supported anymore, please use FSDP.")

    if use_meta_device:
        # allocate memory
        model.to_empty(device=device)

        # Use same seed to init all models
        # Intuitively, one would assume that using the same seed would result in sharded parameters
        # having the same values across their sharded parts, e.g. a parameter of shape [512, 1k] that
        # gets sharded into A[256, 1k] and B[256, 1k] would result in A == B.
        # However, the initilization does not know how memory is stored, and just see a tensor C (A + B)
        # and initializes C independently on how it is stored in memory.
        # This means that A != B and we can use this trick to initialize the model irrespectively of it is distributed.
        seed = exp_cfg["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)  # noqa: NPY002

        # initialize model's parameters
        init_parameters(model, verbose=exp_cfg.get("verbose", False))

        # revert back to per-model seed
        seed = exp_cfg["seed"] + model_rank
        torch.manual_seed(seed)
        np.random.seed(seed)  # noqa: NPY002

    if wandb.run is not None and cfg["wandb"]["watch_freq"] is not None:
        wandb.watch(model, log_freq=cfg["wandb"]["watch_freq"])

    # TODO: be more transparent about this, either via honing the config or warning, etc.
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported() and exp_cfg["dtype"] == "bfloat16"
        else torch.float16
    )
    logger.info(f"Dtype={dtype}")

    optimizer, n_parameters = create_optimizer(model, cfg["optim"])

    loss_scaler = optim_scaler.NativeScaler(enabled=dtype == torch.float16)

    total_iters = exp_cfg["total_iterations"]

    # Preprocessing schedulers lengths to convert them from absolute to relative if needed.
    # In the config file, the sum of lengths for the scheduler should add up to the total
    # number of iterations in the experiment config if absolute values are set for the lengths.
    if "lengths" in cfg["optim"]["scheduler"]:
        total_length = np.sum(cfg["optim"]["scheduler"]["lengths"])
        new_lengths = [float(length) for length in np.asarray(cfg["optim"]["scheduler"]["lengths"]) / total_length]
        OmegaConf.update(cfg, "optim.scheduler.lengths", new_lengths)

        # If optim.scheduler.absolute_length flag is set we check of the lengths add up to
        # total iterations specified in the config.
        if cfg["optim"]["scheduler"].get("absolute_lengths", False):
            assert total_iters == total_length, (
                f"total sum of lengths ({total_length}) and total iters ({total_iters})should be the same."
            )
            cfg["optim"]["scheduler"].pop("absolute_lengths")
        # Done processing and updating the scheduler lengths.

    if cfg.optim.scheduler_type == "torch":
        lr_scheduler = instantiate(cfg["optim"]["scheduler"])(optimizer=optimizer, T_max=total_iters)
    else:
        lr_scheduler = instantiate(cfg["optim"]["scheduler"])

    criterion = instantiate(cfg["loss"])

    if exp_cfg["resume"]:
        if exp_cfg["eval"]:
            assert not model_cfg["checkpoint"], (
                "Both resume and checkpoint were given during eval mode. Please provide only one of them."
            )

        logger.info(f"Resuming from checkpoint at path: {exp_cfg['resume']}")
        # this function loads the checkpoint in-place
        # it also updates the "start_iteration" in the exp_cfg
        # this is an ugly fix for now that will likely change once we improve our dataloaders.
        DistributedCheckpointUtils.resume_training(
            cfg=cfg,
            exp_cfg=exp_cfg,
            local_rank=rank_info.local_rank,
            model=model,
            optimizer=optimizer,
            data_loader_train=data_loader_train,
            lr_scheduler=lr_scheduler,
            loss_scaler=loss_scaler,
        )

        if not isinstance(data_loader_train, torch.utils.data.DataLoader):
            data_loader_train = instantiate(data_cfg["train"]["dataloader"])(
                dataset=dataset_train,
                sampler=sampler_train,
                collate_fn=collate_fn,
                iteration=exp_cfg["start_iteration"],
            )

    non_compiled_model = model
    if exp_cfg["torch_compile"]:
        model = torch.compile(model)
        logger.info("Model running with torch.compile enabled.")

    if cfg["val_loss"] is None:
        val_criterion = criterion
    else:
        val_criterion = instantiate(cfg["val_loss"])

    metrics_computer = instantiate(cfg["metrics"])

    if exp_cfg["eval"]:
        for dataset_name, data_loader_val in data_loaders_val.items():
            _test_stats = evaluate(
                model=non_compiled_model,
                data_loader=data_loader_val,
                device=device,
                iteration=0,
                criterion=val_criterion,
                metrics_computer=metrics_computer,
                tokenizer=validation_tokenizer,
                dtype=dtype,
                dataset_name=dataset_name,
            )

        return

    logger.info(f"Start training for {exp_cfg['total_iterations']} iterations")
    start_time = time.time()
    evaluator_callback = partial(
        maybe_eval_and_save_ckpt,
        cfg=cfg,
        model=non_compiled_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_scaler=loss_scaler,
        data_loader_train=data_loader_train,
        data_loaders_val=data_loaders_val,
        device=device,
        n_parameters=n_parameters,
        total_iters=total_iters,
        criterion=val_criterion,
        metrics_computer=metrics_computer,
        tokenizer=validation_tokenizer,
        dtype=dtype,
        generation_kwargs=data_cfg["generation_kwargs"],
    )

    _train_stats = train(
        model=model,
        criterion=criterion,
        data_loader=data_loader_train,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        total_iters=total_iters,
        evaluator_callback=evaluator_callback,
        loss_scaler=loss_scaler,
        max_norm=cfg["optim"]["grad_clip"],
        set_training_mode=True,
        dtype=dtype,
        amp_enabled=exp_cfg["amp_enabled"],
        test_frequency=exp_cfg["test_frequency"],
        ckpt_save_freq=exp_cfg["ckpt_save_freq"],
        start_iteration=exp_cfg["start_iteration"],
        gradient_accumulation_steps=cfg["optim"]["gradient_accumulation_steps"],
        enable_loss_parallel=exp_cfg.get("enable_loss_parallel", False),
        fsdp_extras=fsdp_extras,
    )

    _ = evaluator_callback(iteration=total_iters)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
