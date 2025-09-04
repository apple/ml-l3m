# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import logging
from collections.abc import Callable
from functools import reduce
from itertools import chain
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.tensor
import torch.nn as nn

from l3m.constants.typing import DATA_DICT
from l3m.model.meta_models import ReadWriteClass

__all__ = [
    "move_to_device",
    "get_chunk_from_elem",
    "chunk_batch",
    "RandomBinaryChoice",
    "get_module",
    "set_module",
    "has_module",
    "init_parameters",
]


logger = logging.getLogger("l3m")


def move_to(obj: Any, device: torch.device | str) -> Any:
    """Moves object to the specified Pytorch device if possible.

    Args:
        obj: Object to move to Pytorch device.
        device: Pytorch device or a string representing the device.
    """

    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    elif hasattr(obj, "to"):
        # for things that are not tensors but contain .to(device)
        # e.g. the BlockMask for flexattention
        return obj.to(device)
    return obj


def move_to_device(data_dict: DATA_DICT, device: torch.device | str) -> Any:
    """Moves all elements in the dict to the specified Pytorch device when possible.

    Args:
        data_dict: Dictionary containing elements to move to device.
        device: Pytorch device or a string representing the device.
    """

    if isinstance(data_dict, dict):
        data_ = {}
        for k, v in data_dict.items():
            data_[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        return data_
    elif isinstance(data_dict, tuple):
        return (move_to(e, device=device) for e in data_dict)
    else:
        raise ValueError(f"batch type is not supported: {type(data_dict)}")


def get_chunk_from_elem(elem: Any, chunk_id: int, num_chunks: int) -> Any:
    """Gets a chunk of an element based on the chunk ID and number of chunks.

    Args:
        elem: The element to chunk.
        chunk_id: The ID of the chunk to retrieve.
        num_chunks: The total number of chunks.

    Returns:
        The chunk of the element.
    """

    if not isinstance(elem, torch.Tensor) or num_chunks == 1:
        # not all objects support slicing, e.g flex_attention.block_mask objects.
        return elem
    assert len(elem) % num_chunks == 0, "Batch size should be divisible by `gradient_accumulation_steps`."
    start = (len(elem) // num_chunks) * chunk_id
    end = (len(elem) // num_chunks) * (chunk_id + 1)
    return elem[start:end]


def chunk_batch(batch: DATA_DICT, num_chunks: int) -> list[DATA_DICT]:
    """Chunks a batch into a list of smaller batches.

    Args:
        batch: The batch to chunk.
        num_chunks: The number of chunks to create.

    Returns:
        A list of chunked batches.
    """

    chunked_batch = []
    for i in range(num_chunks):
        chunked_batch.append({key: get_chunk_from_elem(value, i, num_chunks) for key, value in batch.items()})
    return chunked_batch


class RandomBinaryChoice(ReadWriteClass):
    def __init__(self, prob: float, seed: int = 0, cast_type: Callable = int, **kwargs: Any):
        super().__init__(**kwargs)
        self.prob = prob
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.cast_type = cast_type

    def __call__(self) -> dict[str, Any]:
        return self.cast_type(self.rng.random() < self.prob)


def get_module(model: nn.Module, access_string: str) -> nn.Module:
    """Get a specific part of the model given a pytorch-style model name string, e.g., 'model.backbone.layer.0'

    Args:
        model: Pytorch model.
        access_string: String representing module to access.

    Returns:
        The desired Pytorch module.
    """

    names = access_string.split(sep=".")
    return reduce(getattr, names, model)


def set_module(model: nn.Module, access_string: str, value: nn.Module) -> None:
    """Modify a specific part of the model given a pytorch-style model name string, e.g., 'model.backbone.layer.0'.

    Args:
        model: Pytorch model.
        access_string: String representing module to access.
        value: New value for that Pytorch module.

    Returns:
        The desired Pytorch module.
    """
    names = access_string.split(sep=".")
    parent = reduce(getattr, names[:-1], model)
    setattr(parent, names[-1], value)


def has_module(model: nn.Module, access_string: str) -> bool:
    """Checks if a module is present in the model.

    Args:
        model: Pytorch model.
        access_string: String representing module to access.
    """

    try:
        get_module(model, access_string)
        return True
    except AttributeError:
        return False


def init_parameters(model: nn.Module, verbose: bool = False) -> None:
    """Initialize the model's parameters following an order of modules.

    This function will follow the initialization order to allow proper overriding of the inits.
    Assuming your model is:

    .. code-block:: python

        Model(
            module1(
                linear1: nn.Linear(...),
                linear2: nn.Linear(...),
            ),
            module2(
                block1: Block(
                    linear1: nn.Linear(...),
                    norm1: nn.LayerNorm(...),
                ),
                block2: Block(
                    linear1: nn.Linear(...),
                    norm1: nn.LayerNorm(...),
                )
            ),
            linear: nn.Linear(...),
        )

    The execution order will be:

    1. linear
    2. module2.block2.norm1, module2.block2.linear, module2.block2
    3. module2.block1.norm1, module2.block1.linear, module2.block1
    4. module2
    5. module1.linear2, module1.linear1
    6. module1

    Args:
        model: Full Pytorch model to initialize.
        verbose: Whether to be verbose.
    """

    # these are hard-coded and kept consistent across L3M/Pytorch
    custom_init_function = "init_weights"
    pytorch_init_function = "reset_parameters"

    initialization_order = [name for name, _ in model.named_modules()]
    initialization_order = list(reversed(initialization_order))

    # manually init everything with -2 to allow us to track if params got initialized
    for p in chain(model.parameters(), model.buffers()):
        p.data.fill_(-2.0)

    if verbose:
        logger.info(f"Initialization order of modules: {initialization_order}")
    needs_init = [name for name, _ in model.named_parameters()] + [name for name, _ in model.named_buffers()]
    for module_path in initialization_order:
        # main model, either MetaModel or InSeriesMetaModel
        if module_path == "":
            module = model
        else:
            module = get_module(model, module_path)

        # collect parameters and buffers before init
        params_and_buffers = {
            name: (p.to_local().clone() if isinstance(p, torch.distributed.tensor.DTensor) else p.clone())
            for name, p in module.named_parameters()
        } | {
            name: (p.to_local().clone() if isinstance(p, torch.distributed.tensor.DTensor) else p.clone())
            for name, p in module.named_buffers()
        }

        if hasattr(module, custom_init_function):
            module.init_weights()
            if verbose:
                logger.info(f"Calling {custom_init_function} in {module_path}")
        elif hasattr(module, pytorch_init_function):
            module.reset_parameters()
            if verbose:
                logger.info(f"Calling {pytorch_init_function} in {module_path}")
        else:
            if verbose:
                logger.info(f"Skipping {module_path}")

        # check which parameters changed
        for name, p in chain(module.named_parameters(), module.named_buffers()):
            parameter_path = f"{module_path}.{name}"

            if isinstance(p, torch.distributed.tensor.DTensor):
                p = p.to_local()
                # some parameters might not be present in all devices
                # because they are too small, e.g. positional embedding or cls token
                # check it here and account as it was initialized
                if len(p) == 0 and parameter_path in needs_init:
                    needs_init.remove(parameter_path)

            # keep track if the parameter changed
            # we don't care how many times a parameter get's reinitialized
            if parameter_path in needs_init and torch.any(p != params_and_buffers[name]):
                needs_init.remove(parameter_path)

    if dist.is_initialized():
        dist.barrier()

    # if there are any parameter/buffer that did not change, something is wrong
    assert not needs_init, (
        "Some parameters did not get initialized. "
        "Make sure every parameter/buffer can be initialized by calling init_weights() or reset_parameters().\n"
        f"Parameters/buffers: {needs_init}"
    )
