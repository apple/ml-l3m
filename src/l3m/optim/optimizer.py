# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""Adapted from:

- ELECTRA https://github.com/google-research/electra
- BEiT: https://github.com/microsoft/unilm/tree/master/beit
- MAE: https://github.com/facebookresearch/mae/blob/main/util/lr_decay.py
"""

import json
import logging
from fnmatch import fnmatch
from typing import Any

from hydra.utils import instantiate
from torch import nn, optim

from l3m.constants.generic import HPARAM_PREFIX
from l3m.optim import mup

_OptimParams = list[dict[str, Any]]
_NoWDNames = list[str]

__all__ = [
    "create_optimizer",
    "get_param_groups",
    "get_param_groups_with_lr_scalers",
    "get_param_groups_with_lr_wd_grid",
    "param_groups_lrd",
    "get_layer_id_for_vit",
    "get_probe_param_groups",
]

logger = logging.getLogger("l3m")


def create_optimizer(model: nn.Module, cfg_optim: dict[str, Any]) -> tuple[optim.Optimizer, int]:
    """Create an optimizer for a model given a config.

    Args:
        model: model for which to create the optimizer.
        cfg_optim: dictionary containing the following values:

            - ``'optimizer'`` - optimizer to :func:`~hydra.utils.instantiate`.
            - ``'weight_decay'`` - weight decay.
            - ``'freeze_list'`` - list of patterns for :func:`~fnmatch.fnmatch` to
              determine which parameters to freeze.
            - ``'unfreeze_list'`` - list of patterns to determine which parameters to
              unfreeze. Always applied after the ``'freeze_list'``.
            - ``'wd_exclude'`` - list patterns to determine parameters where weight
              decay is **not** applied.
            - ``'lr_decay'`` - learning rate decay across layers.
            - ``'probe_hparam_search'`` - flag to indicate if the learning rate should
              be deduced from the module names in the model head.

    Returns:
        The optimizer and the number of trainable parameters.
    """

    if cfg_optim["freeze_list"] is not None:
        for n, p in model.named_parameters():
            if any(fnmatch(n, excl) for excl in cfg_optim["freeze_list"]):
                p.requires_grad = False

    if cfg_optim["unfreeze_list"] is not None:
        for n, p in model.named_parameters():
            if any(fnmatch(n, excl) for excl in cfg_optim["unfreeze_list"]):
                p.requires_grad = True

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_parameters_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    frozen_parameter_names = [n for n, p in model.named_parameters() if not p.requires_grad]

    logger.info(f"Number of trainable params: {n_parameters}.")
    logger.info(f"Number of frozen params: {n_parameters_frozen}.")
    logger.info(f"Frozen params: {frozen_parameter_names}.")

    # Create parameter groups based on configuration
    if cfg_optim.get("mup", False):
        mup_cfg = cfg_optim["mup"]

        # Use MUP initialization if specified
        if cfg_optim["mup"].get("initialize", False):
            mup.mup_init(model, mup_cfg)

        optim_params, no_wd_names = mup.create_mup_param_groups(
            model,
            mup_cfg=mup_cfg,
            weight_decay=cfg_optim["optimizer"]["weight_decay"],
            wd_exclude_list=cfg_optim["wd_exclude"],
        )

    elif cfg_optim["probe_hparam_search"]:
        optim_params, no_wd_names = get_probe_param_groups(
            model,
            base_lr=cfg_optim["optimizer"]["lr"],
            weight_decay=cfg_optim["optimizer"]["weight_decay"],
            wd_exclude_list=cfg_optim["wd_exclude"],
        )

    else:
        optim_params, no_wd_names = param_groups_lrd(
            model,
            cfg_optim["optimizer"]["weight_decay"],
            cfg_optim["wd_exclude"],
            cfg_optim["lr_decay"],
            cfg_optim["lr_scalers"],
        )

    logger.info(f"Parameters without weight decay:\n {no_wd_names}")

    optimizer = instantiate(cfg_optim["optimizer"])(params=optim_params)

    return optimizer, n_parameters


def get_param_groups(
    model: nn.Module,
    weight_decay: float = 0.05,
    wd_exclude_list: list[str] = None,
) -> tuple[_OptimParams, _NoWDNames]:
    """Creates parameter groups for the optimizer, applying weight decay to some parameters, and excluding others.

    Args:
        model: The model whose parameters are being grouped.
        weight_decay: The weight decay to apply to the parameters
            that are not excluded.
        wd_exclude_list: A list of strings used as patterns
            to match parameter names that should be excluded from weight decay.

    Returns:
        A tuple containing:

        - A list of dictionaries, where each dictionary defines a parameter group for the optimizer.
        - A list of names of the parameters that were excluded from weight decay.
    """

    wd_exclude_list = wd_exclude_list or []
    no_wd_names = []
    p_wd, p_non_wd = [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if any(fnmatch(n, excl) for excl in wd_exclude_list):
            p_non_wd.append(p)
            no_wd_names.append(n)
        else:
            p_wd.append(p)

    optim_params = [
        {"params": p_wd, "weight_decay": weight_decay},
        {"params": p_non_wd, "weight_decay": 0},
    ]

    return optim_params, no_wd_names


def get_param_groups_with_lr_scalers(
    model: nn.Module,
    lr_scalers: dict[str, float],
    weight_decay: float = 0.05,
    wd_exclude_list: list[str] = None,
):
    """Creates parameter groups for the optimizer, applying different learning rate
    scalers and weight decay settings to different parameters.

    Args:
        model: The model whose parameters are being grouped.
        lr_scalers: A dictionary where keys are parameter
            name patterns and values are learning rate scaling factors.
        weight_decay: The weight decay to apply to the parameters
            that are not excluded.
        wd_exclude_list: A list of strings used as patterns
            to match parameter names that should be excluded from weight decay.

    Returns:
        A tuple containing:

        - A list of dictionaries, where each dictionary defines a parameter group for the optimizer.
        - A list of names of the parameters that were excluded from weight decay.
    """

    wd_exclude_list = wd_exclude_list or []
    param_group_names = {}
    param_groups = {}
    no_wd_names = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        p_name = "non_scaled"
        for lr_scaler_name in lr_scalers.keys():
            if fnmatch(n, f"*{lr_scaler_name}*"):
                p_name = lr_scaler_name
                break

        if any(fnmatch(n, excl) for excl in wd_exclude_list):
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        group_name = f"{p_name}_{g_decay}"
        if group_name not in param_group_names:
            lr_scale = 1.0 if p_name == "non_scaled" else lr_scalers[p_name]

            param_group_names[group_name] = {
                "lr_scale": lr_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": lr_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    logger.info(f"Param groups = {json.dumps(param_group_names, indent=2)}")
    return list(param_groups.values()), no_wd_names


def get_param_groups_with_lr_wd_grid(
    model: nn.Module,
    lr_wd_grid: dict[str, tuple[float, float]],
    wd_exclude_list: list[str] = None,
):
    """Creates parameter groups for the optimizer, applying different learning rate
    scalers and weight decay values based on a grid.

    Args:
        model: The model whose parameters are being grouped.
        lr_wd_grid: A dictionary where keys
            are parameter name patterns and values are tuples of (learning rate scaling factor, weight decay value).
        wd_exclude_list: A list of strings used as
            patterns to match parameter names that should be excluded from weight decay.

    Returns:
        A tuple containing:

        - A list of dictionaries, where each dictionary defines a parameter group for the optimizer.
        - A list of names of the parameters that were excluded from weight decay.
    """

    wd_exclude_list = wd_exclude_list or []
    param_group_names = {}
    param_groups = {}
    no_wd_names = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        for p_name, (lr_scale, this_decay) in lr_wd_grid.items():  # noqa: B007
            if fnmatch(n, f"*{p_name}*"):
                break
        else:
            p_name = "non_scaled"
            lr_scale = 1.0
            this_decay = 0.0

        if any(fnmatch(n, excl) for excl in wd_exclude_list):
            g_decay = "no_decay"
        else:
            g_decay = "decay"

        group_name = f"{p_name}_{g_decay}"
        if group_name not in param_group_names:
            param_group_names[group_name] = {
                "lr_scale": lr_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": lr_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    logger.info(f"Param groups = {json.dumps(param_group_names, indent=2)}")
    return list(param_groups.values()), no_wd_names


def param_groups_lrd(
    model: nn.Module,
    weight_decay: float = 0.05,
    wd_exclude_list: list[str] = None,
    layer_decay: float | None = None,
    lr_scalers: dict[str, float] | None = None,
) -> tuple[_OptimParams, _NoWDNames]:
    """Creates parameter groups for the optimizer, applying layer-wise learning
    rate decay (LRD) and/or learning rate scalers, and excluding certain parameters from weight decay.

    Args:
        model: The model whose parameters are being grouped.
        weight_decay: The weight decay to apply to the parameters
            that are not excluded.
        wd_exclude_list: A list of strings used as patterns to
            match parameter names that should be excluded from weight decay.
        layer_decay: The layer decay rate.  If provided, learning
            rates will be scaled layer-wise based on this decay rate.
        lr_scalers: A dictionary where keys are parameter
            name patterns and values are learning rate scaling factors.
            If provided, learning rates will be scaled based on these factors.

    Returns:
        A tuple containing:

        - A list of dictionaries, where each dictionary defines a parameter group for the optimizer.
        - A list of names of the parameters that were excluded from weight decay.
    """

    wd_exclude_list = wd_exclude_list or []

    if lr_scalers is not None:
        return get_param_groups_with_lr_scalers(
            model,
            lr_scalers,
            weight_decay,
            wd_exclude_list,
        )
    if layer_decay is None:
        return get_param_groups(
            model,
            weight_decay,
            wd_exclude_list,
        )

    param_group_names = {}
    param_groups = {}

    no_wd_names = []
    num_layers = len(model.trunk.blocks) + 1

    layer_scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if any(fnmatch(n, excl) for excl in wd_exclude_list):
            g_decay = "no_decay"
            this_decay = 0.0
            no_wd_names.append(n)
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = f"layer_{layer_id}_{g_decay}"

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values()), no_wd_names


def get_layer_id_for_vit(name: str, num_layers: int) -> int:
    name = name.replace("_fsdp_wrapped_module.", "")
    if name.startswith("preprocessor"):
        return 0
    elif name.startswith("trunk.blocks"):
        return int(name.split(".")[2]) + 1
    else:
        return num_layers


def get_probe_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float = 0.0,
    wd_exclude_list: list[str] = None,
) -> tuple[_OptimParams, _NoWDNames]:
    """Creates parameter groups for the optimizer specifically
    for probe heads, using learning rates and weight decays specified in the head names.

    Args:
        model: The model whose parameters are being grouped.
            Assumes the model has a `head` attribute with `heads` as a
            dictionary where keys are the head names.
        base_lr: The base learning rate.  Learning rates for each
            head are scaled relative to this value.
        weight_decay: The default weight decay to use
            if a specific weight decay is not found in the head name.
        wd_exclude_list: A list of strings used as patterns
            to match parameter names that should be excluded from weight decay.

    Returns:
        A tuple containing:

        - A list of dictionaries, where each dictionary defines a parameter group for the optimizer.
        - A list of names of the parameters that were excluded from weight decay.
    """
    wd_exclude_list = wd_exclude_list or []
    lr_wd_grid = {}

    for head_name in model.head.heads.keys():
        assert head_name.startswith(HPARAM_PREFIX), head_name
        # <prefix>_LR_<val>_...
        _, _, lr, _, _, *wd = head_name.split("_")
        if len(wd):
            _, wd = wd
            wd = float(wd.replace(",", "."))
        else:
            wd = weight_decay

        lr_scale = float(lr.replace(",", "."))
        lr_wd_grid[head_name] = (lr_scale / base_lr, wd)

    return get_param_groups_with_lr_wd_grid(model, lr_wd_grid, wd_exclude_list=wd_exclude_list)
