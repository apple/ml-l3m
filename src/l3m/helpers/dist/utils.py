# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import logging
import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.distributed._tensor import DTensor, Replicate
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import distribute_tensor

__all__ = [
    "setup_for_distributed",
    "is_dist_avail_and_initialized",
    "get_world_size",
    "get_rank",
    "get_local_rank",
    "is_main_process",
    "init_distributed_mode",
    "DeviceMeshHandler",
    "aggregate_tensor_across_devices",
    "replicate_tensor_if_distributed",
    "tensor_to_dtensor",
]


logger = logging.getLogger("l3m")


def setup_for_distributed(is_master: bool) -> None:
    """Disables printing when not in master process"""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0

    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = dist.get_rank() % torch.cuda.device_count()
    return rank


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed_mode(cfg: DictConfig) -> None:
    """Initializes the distributed environment for an interactive node.

    Args:
        cfg: Experiment config.
    """

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        cfg["experiment"]["rank"] = int(os.environ["RANK"])
        cfg["experiment"]["world_size"] = int(os.environ["WORLD_SIZE"])
        cfg["experiment"]["gpu"] = int(os.environ["LOCAL_RANK"])
    else:
        logger.info("Not using distributed mode")
        cfg["experiment"]["distributed"] = False
        return

    cfg["experiment"]["distributed"] = True

    torch.cuda.set_device(cfg["experiment"]["gpu"])  # set device based on local rank
    cfg["experiment"]["dist_backend"] = "nccl"
    logger.info(
        "| distributed init (rank {}): {}".format(cfg["experiment"]["rank"], "env://"),
    )
    dist.init_process_group(
        backend=cfg["experiment"]["dist_backend"],
        init_method="env://",
        world_size=cfg["experiment"]["world_size"],
        rank=cfg["experiment"]["rank"],
        timeout=timedelta(minutes=cfg["experiment"]["nccl_timeout_mins"]),
    )
    dist.barrier()
    setup_for_distributed(cfg["experiment"]["rank"] == 0)


@dataclass
class RankInfo:
    local_rank: int  # pytorch local rank, unique per device
    global_rank: int  # pytorch rank, unique per device
    world_size: int  # distributed work size
    model_rank: int  # model id (will be unique per device if not doing TP/PP/CP)
    cp_rank: int | None = None  # cp rank of that device within the context parallel unit
    tp_rank: int | None = None  # tp rank of that device within the tp unit
    pp_rank: int | None = None  # pp rank of that device within the pipeline parallel unit


class DeviceMeshHandler:
    """A class to manage the device mesh for distributed training.

    This class provides methods for initializing, accessing, and querying the device mesh,
    including checking for data parallelism (DP), tensor parallelism (TP),
    context parallelism (CP), and pipeline parallelism (PP).
    """

    device_mesh: DeviceMesh = None

    @classmethod
    def has_device_mesh(cls) -> bool:
        return cls.device_mesh is not None

    @classmethod
    def get_device_mesh(cls, exp_cfg: DictConfig) -> DeviceMesh | None:
        # if we already have a device mesh set up
        if cls.device_mesh is not None:
            return cls.device_mesh

        # there's no device mesh
        if exp_cfg["fsdp"] is None:
            return None

        fsdp_cfg = exp_cfg["fsdp"]
        fsdp_type = fsdp_cfg["sharding_strategy"].lower()
        world_size = exp_cfg["world_size"]

        cp_degree = fsdp_cfg.get("cp_degree", 1)
        tp_size = fsdp_cfg.get("tp_size", 1)
        if fsdp_type == "no_shard":
            dp_shard = 1
            dp_replicate = world_size // (tp_size * cp_degree)
        else:
            if "dp_replicate" in fsdp_cfg:
                dp_shard = fsdp_cfg.get("dp_shard", 1)
                dp_replicate = fsdp_cfg.get("dp_replicate", 1)
            else:
                dp_shard = world_size // (tp_size * cp_degree)
                dp_replicate = 1

        assert dp_replicate * dp_shard * tp_size * cp_degree == world_size, (
            "dp_replicate * dp_shard * cp_degree * tp_size "
        )
        f"({dp_replicate} * {dp_shard} * {cp_degree} * {tp_size}) != world_size ({world_size})"

        dims = [dp_replicate, dp_shard]
        names = ["dp_replicate", "dp_shard"]
        if cp_degree > 1:
            dims.append(cp_degree)
            names.append("cp")
        if tp_size > 1:
            dims.append(tp_size)
            names.append("tp")
        dims = tuple(dims)
        names = tuple(names)

        cls.device_mesh = init_device_mesh("cuda", mesh_shape=dims, mesh_dim_names=names)

        logger.info(f"Created world mesh: {cls.device_mesh}")
        return cls.device_mesh

    @classmethod
    def get_rank_info(cls) -> RankInfo:
        device_mesh = cls.device_mesh

        local_rank = get_local_rank()
        global_rank = get_rank()
        if device_mesh is None:
            logger.info(
                "No device_mesh available. If device_mesh was suppose to "
                "be available, make sure to call get_device_mesh() before.",
            )
            model_rank = global_rank
            world_size = get_world_size()
            cp_rank = None
            tp_rank = None

        else:
            replicate_mesh = device_mesh["dp_replicate"]
            shard_mesh = device_mesh["dp_shard"]
            # number of replicate devices
            world_size = replicate_mesh.size()
            model_rank = replicate_mesh.get_local_rank()
            if shard_mesh.size() > 1:
                # multiply with number of shard device
                model_rank = model_rank * world_size + shard_mesh.get_local_rank()
                world_size *= shard_mesh.size()

            cp_rank = None
            if cls.cp_enabled():
                cp_rank = device_mesh["cp"].get_local_rank()
            tp_rank = None
            if cls.tp_enabled():
                tp_rank = device_mesh["tp"].get_local_rank()

        return RankInfo(
            local_rank=local_rank,
            global_rank=global_rank,
            world_size=world_size,
            model_rank=model_rank,
            cp_rank=cp_rank,
            tp_rank=tp_rank,
        )

    @classmethod
    def dp_enabled(cls) -> bool:
        return cls.dp_replicate_enabled() or cls.dp_shard_enabled()

    @classmethod
    def dp_replicate_enabled(cls) -> bool:
        return (
            cls.device_mesh is not None
            and "dp_replicate" in cls.device_mesh.mesh_dim_names
            and cls.device_mesh["dp_replicate"].size() > 1
        )

    @classmethod
    def dp_shard_enabled(cls) -> bool:
        return (
            cls.device_mesh is not None
            and "dp_shard" in cls.device_mesh.mesh_dim_names
            and cls.device_mesh["dp_shard"].size() > 1
        )

    @classmethod
    def cp_enabled(cls) -> bool:
        return (
            cls.device_mesh is not None and "cp" in cls.device_mesh.mesh_dim_names and cls.device_mesh["cp"].size() > 1
        )

    @classmethod
    def tp_enabled(cls) -> bool:
        return (
            cls.device_mesh is not None and "tp" in cls.device_mesh.mesh_dim_names and cls.device_mesh["tp"].size() > 1
        )

    @classmethod
    def pp_enabled(cls) -> bool:
        return (
            cls.device_mesh is not None and "pp" in cls.device_mesh.mesh_dim_names and cls.device_mesh["pp"].size() > 1
        )


def aggregate_tensor_across_devices(
    x: torch.Tensor | DTensor,
) -> list[torch.Tensor]:
    """Aggregates a tensor across all devices using all_gather.

    Args:
        x: The tensor to aggregate. If it's a DTensor, it will be converted to a local tensor first.

    Returns:
        A list containing the gathered tensors from all devices.
    """

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, device="cuda")
    elif isinstance(x, DTensor):
        x = x.to_local()
    L = [torch.zeros_like(x) for i in range(dist.get_world_size())]
    dist.all_gather(L, x)
    return L


def replicate_tensor_if_distributed(
    source_tensor: torch.Tensor,
    reference_tensor: torch.Tensor | DTensor,
) -> torch.Tensor | DTensor:
    """Converts a PyTorch tensor to a distributed tensor (DTensor) with replicated placement.

    The conversion occurs only if the reference tensor is a DTensor.

    Args:
        source_tensor: The tensor to potentially replicate.
        reference_tensor: The reference tensor. If it's a DTensor, the source tensor will be replicated.

    Returns:
        The replicated tensor (DTensor) or the original tensor if the reference tensor is not a DTensor.
    """
    # if matching tensor is not dtensor, just return the original tensor
    if not isinstance(reference_tensor, DTensor):
        return source_tensor

    return distribute_tensor(
        source_tensor,
        device_mesh=reference_tensor.device_mesh,
        placements=(Replicate(),),
    )


def tensor_to_dtensor(
    source_tensor: torch.Tensor,
    reference_tensor: torch.Tensor | DTensor,
) -> torch.Tensor | DTensor:
    """Converts a PyTorch tensor to a distributed tensor (DTensor) using the same placements as a reference tensor.

    Args:
        source_tensor: The tensor to convert.
        reference_tensor: The reference tensor. Its device mesh and placements will be used for the conversion.

    Returns:
        The converted DTensor, or the original tensor if the reference tensor is not a DTensor.
    """

    # if matching tensor is not dtensor, just return the original tensor
    if not isinstance(reference_tensor, DTensor):
        return source_tensor

    return distribute_tensor(
        source_tensor,
        device_mesh=reference_tensor.device_mesh,
        placements=reference_tensor.placements,
    )
