# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import torch
import torch.distributed as dist

from l3m.helpers.dist import utils as dist_utils

__all__ = ["all_gather_batch", "all_gather_batch_with_grad"]


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all workers with support for backward propagation.

    This implementation does not cut the gradients as torch.distributed.all_gather does.

    Adapted from: https://github.com/facebookresearch/SLIP/blob/main/utils.py
    """

    @staticmethod
    def forward(ctx, x) -> tuple[torch.Tensor, ...]:
        rank_info = dist_utils.DeviceMeshHandler.get_rank_info()
        world_size = rank_info.world_size
        output = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads) -> torch.Tensor:
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def all_gather_batch(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """Performs all_gather operation on the provided tensors.

    Adapted from: https://github.com/facebookresearch/SLIP/blob/main/utils.py

    Args:
        tensors: List of tensors to gather.

    Returns:
        List of gathered tensors.
    """

    world_size = dist_utils.DeviceMeshHandler.get_rank_info().world_size
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_all, tensor, async_op=False)  # performance opt

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))

    return output_tensor


def all_gather_batch_with_grad(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """Performs all_gather operation on the provided tensors keeping
    the graph connected for backward grad computation.

    Adapted from: https://github.com/facebookresearch/SLIP/blob/main/utils.py

    Args:
        tensors: List of tensors to gather.

    Returns:
        List of gathered tensors, with gradient support.
    """

    world_size = dist_utils.DeviceMeshHandler.get_rank_info().world_size

    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))

    return output_tensor
