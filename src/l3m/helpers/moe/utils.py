# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import importlib.util
import inspect
from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "is_megablocks_available",
    "get_public_classes_in_module",
    "get_moe_layers_in_model",
    "is_model_moe",
    "extract_moe_arguments_from_model",
    "get_load_balancing_loss_given_moe",
]


def is_megablocks_available() -> bool:
    return importlib.util.find_spec("megablocks") is not None


def get_public_classes_in_module(module: Any) -> list[tuple[str, Any]]:
    """Retrieve a list of public classes defined in the specified module.

    Args:
        module: The module to inspect.

    Returns:
        Tuple contains the class name and the class itself.
    """

    modules = inspect.getmembers(module, inspect.isclass)
    modules = [(m_name, m_cls) for (m_name, m_cls) in modules if m_name in module.__all__]
    return modules


def get_moe_layers_in_model(model: nn.Module) -> list[nn.Module]:
    """Identify Mixture-of-Experts (MoE) layers within the given model.

    Args:
        model: The model to inspect for MoE layers.

    Returns:
        A list of MoE layers found in the model.
    """
    from l3m.model.layers import moe as l3m_moe_module

    _, moe_classes = zip(*get_public_classes_in_module(l3m_moe_module), strict=True)
    moe_layers = [m for m in model.modules() if isinstance(m, moe_classes)]
    return moe_layers


def is_model_moe(model: nn.Module) -> bool:
    """Determine if the specified model contains any MoE layers.

    Args:
        model: The model to check.

    Returns:
        True if the model contains MoE layers, False otherwise.
    """
    return len(get_moe_layers_in_model(model)) > 0


def extract_moe_arguments_from_model(model: nn.Module) -> list[dict[str, Any]]:
    """Extract the arguments of MoE layers within the given model.

    Args:
        model: The model to extract MoE arguments from.

    Returns:
        A list of arguments from each MoE layer.
    """
    moe_args = [moe_layer.megablocks_arguments for moe_layer in get_moe_layers_in_model(model)]
    return moe_args


def get_load_balancing_loss_given_moe(model: nn.Module) -> torch.Tensor:
    """Calculate the load balancing loss for a given Mixture-of-Experts (MoE) model.

    This function extracts the MoE arguments from the model, computes the load balancing
    loss using those arguments for the first MoE layer, and then clears the computed loss.

    Args:
        model: The MoE model from which to extract arguments.

    Returns:
        The load balancing loss as a float.

    Notes:
        Assumes that all MoE layers within the model have identical configurations.
    """
    assert is_megablocks_available(), "Please install megablocks first"
    from megablocks.layers.moe import batched_load_balancing_loss, clear_load_balancing_loss

    moe_args = extract_moe_arguments_from_model(model)
    moe_arg = moe_args[0]  # assumes that all moe layers have the same config for now
    try:
        lbl_loss = batched_load_balancing_loss(moe_arg)
    except ValueError as e:
        # load balancing loss is recorded only in the training mode, otherwise, the global variable is empty
        if "not enough values to unpack" not in str(e):
            raise
        assert not model.training, "Expected the model to be in evaluation mode."
        return torch.tensor(torch.nan)
    clear_load_balancing_loss()
    return lbl_loss
