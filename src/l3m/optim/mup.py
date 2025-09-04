# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""Maximal Update Parameterization:

- TP5: https://arxiv.org/abs/2203.03466
- MuP-Simple: https://arxiv.org/abs/2309.14322
"""

import logging
import math
from fnmatch import fnmatch
from typing import Any

import torch
import torch.nn as nn

__all__ = ["mup_init", "create_mup_param_groups"]

logger = logging.getLogger("l3m")


def mup_init(model: nn.Module, mup_cfg: dict[str, Any]) -> None:
    """Scale model's parameters in place to achieve a MuP-based initialization.

    Example of a ``mup_cfg``:

    .. code-block:: python

        {
            "base_width": 384,
            "param_groups": [
                {
                    "pattern": "*attn.qkv.weight",
                    "width": 4096,
                    "init_scale_expr": "(base_width / width)"
                },
                {
                    "pattern": ".*linear.weight",
                    "width": 1024,
                    "init_scale_expr": "math.sqrt(base_width) / width"
                }
            ]
        }

    Args:
        model: model whose parameters are to be scaled.
        mup_cfg: config with an example spec above.
    """

    base_width = float(mup_cfg.get("base_width", None))
    assert base_width is not None, "base_width needs to be specified for MuP. Please tune at small scale."
    param_groups_cfg: list[dict[str, Any]] = mup_cfg.get("param_groups", [])

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # find a config entry for this param
        matched_cfg = None
        for pg in param_groups_cfg:
            pattern = pg["pattern"]
            if fnmatch(name, pattern):
                matched_cfg = pg
                break

        if matched_cfg is None:
            # no match => user didn't specify any MuP scaling for param
            continue

        # read 'width' from config
        config_width = matched_cfg.get("width", base_width)
        config_width = float(config_width)

        local_ctx = {"base_width": base_width, "width": config_width}

        # evaluate init_scale
        init_scale_expr = matched_cfg.get("init_scale_expr", "1.0")
        local_ctx["math"] = math  # for ops like math.sqrt(), etc
        try:
            init_scale = eval(init_scale_expr, {}, local_ctx)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[mup_init] init_scale_expr failed for param={name}: {e}")
            init_scale = 1.0

        # scale param to achieve the desired init_scale
        with torch.no_grad():
            param.mul_(init_scale)

        logger.debug(
            f"[mup_init] param={name}, pattern={matched_cfg['pattern']}, width={config_width}, init_scale={init_scale}"
        )


def create_mup_param_groups(
    model: nn.Module,
    mup_cfg: dict[str, Any],
    weight_decay: float = 0.0,
    wd_exclude_list: list[str] = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Create MuP parameter groups for an optimizer.

    Example of a ``mup_cfg``:

    .. code-block:: python

        {
            "base_width": 384,
            "param_groups": [
                {
                    "pattern": "*attn.qkv.weight",
                    "width": 4096,
                    "lr_scale_expr": "(base_width / width)"
                }
            ]
        }

    Then for each named parameter:

    1. find the param-group whose 'pattern' regex matches,
    2. evaluate 'lr_scale_expr' => lr_scale,
    3. set "lr_scale=..." in the param group.

    Args:
        model: model whose parameters are to be grouped.
        mup_cfg: config with example spec above.
        weight_decay: default weight decay for non-excluded params.
        wd_exclude_list: patterns for excluding weight decay.

    Returns:
        A tuple containing:

        - list of optimizer parameter groups.
        - list of parameter names with no weight decay.
    """

    wd_exclude_list = wd_exclude_list or []
    base_width = mup_cfg.get("base_width", None)
    assert base_width is not None, (
        "MuP works by tuning a small model at base_width and then scaling up. "
        "Please specify `base_width` under the `mup` field."
    )
    base_width = float(base_width)
    param_groups_cfg: list[dict[str, Any]] = mup_cfg.get("param_groups", [])

    param_groups: list[dict[str, Any]] = []
    no_wd_names: list[str] = []
    assigned_params = set()

    # Iterate over named_parameters once. For each param, we see which config pattern matches.
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        matched_cfg = None
        for pg in param_groups_cfg:
            pattern = pg["pattern"]
            if fnmatch(name, pattern):
                matched_cfg = pg
                break

        if matched_cfg is None:
            # no config matched => we handle in fallback at the end
            continue

        # read the user-specified 'width' from config
        config_width = matched_cfg.get("width", base_width)
        config_width = float(config_width)

        # Parse the expression for LR scale.
        lr_scale_expr = matched_cfg.get("lr_scale_expr", "1.0")
        # math for ops like math.sqrt(), etc
        local_ctx = {
            "base_width": float(base_width),
            "width": config_width,
            "math": math,
        }

        # Evaluate lr_scale
        try:
            lr_scale = eval(lr_scale_expr, {}, local_ctx)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"[create_mup_param_groups] Could not eval lr_scale_expr={lr_scale_expr} "
                f"for param={name}: {e}. Fallback to 1.0"
            )
            lr_scale = 1.0

        # WD exclude logic
        no_decay = False
        if any(fnmatch(name, excl) for excl in wd_exclude_list):
            no_decay = True

        # add to param groups
        param_groups.append(
            {
                "params": [param],
                "lr_scale": lr_scale,
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
        )

        if no_decay:
            no_wd_names.append(name)

        assigned_params.add(param)

        logger.debug(
            f"[create_mup_param_groups] match name={name}, pattern={matched_cfg['pattern']}, "
            f"width={config_width}, lr_scale={lr_scale}, no_decay={no_decay}"
        )

    # fallback group for unmatched
    remaining_params = []
    remaining_no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad or (param in assigned_params):
            continue

        no_decay = False
        if any(fnmatch(name, excl) for excl in wd_exclude_list):
            no_decay = True

        if no_decay:
            remaining_no_decay.append(param)
            no_wd_names.append(name)
        else:
            remaining_params.append(param)

        logger.debug(f"[create_mup_param_groups] fallback param={name}, no_decay={no_decay}")

    if remaining_params:
        param_groups.append({"params": remaining_params, "weight_decay": weight_decay})

    if remaining_no_decay:
        param_groups.append({"params": remaining_no_decay, "weight_decay": 0.0})

    return param_groups, no_wd_names
