# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import logging
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh

from l3m import constants
from l3m.helpers.dist.cp import build_cp_plan
from l3m.helpers.utils import get_module, set_module

try:
    from l3m.helpers.dist.tp import assert_can_do_tp, tp_parallelize

    tp_supported = True
except ImportError:
    tp_supported = False

FSDP2_CLASS_NAME_PREFIX = "FSDP"

logger = logging.getLogger("l3m")

__all__ = [
    "build_grouping_plan",
    "build_activation_checkpoint_plan",
    "apply_ac_to_module",
    "parallelize_model",
    "apply_ac",
    "init_fsdp2",
]


def build_grouping_plan(model: nn.Module, exp_cfg: DictConfig) -> list[str]:
    """Creates FSDP grouping plan, i.e., the layer names of the individual FSDP units.

    Args:
        model: The model to be sharded.
        exp_cfg: The experiment configuration.

    Returns:
        A list of layer names to be wrapped by FSDP.
    """

    shard_template = exp_cfg["fsdp"].get("shard_template", None)
    layers_to_wrap = exp_cfg["fsdp"].get("fsdp_layers_to_wrap", None)

    plan = []
    # use the explicit sharding template, e.g. model.transformer.blocks
    if shard_template:
        assert not layers_to_wrap, "Use only one style of sharding."

        if not isinstance(shard_template, Iterable):
            shard_template = [shard_template]

        for template in shard_template:
            module = get_module(model, template)
            if hasattr(module, "__len__"):  # we have the name of a iterable layer, e.g. ModuleList
                n = len(module)
                plan.extend([f"{template}.{i}" for i in range(n)])
            else:  # directly have the name of a module
                plan.append(template)

    # infer from the class names the layers to shard e.g. Block
    elif layers_to_wrap:
        # collect the layer names by matching their class names
        for name, module in model.named_modules():
            class_name = module.__class__.__name__
            # remove fsdp class name prefix
            class_name = (
                class_name[len(FSDP2_CLASS_NAME_PREFIX) :]
                if class_name.startswith(FSDP2_CLASS_NAME_PREFIX)
                else class_name
            )
            for layer in layers_to_wrap:
                if class_name == layer:
                    plan.append(name)
                    break
    else:
        raise ValueError("Cannot shard model, missing shard_template and fsdp_layers_to_wrap.")

    return plan


def build_activation_checkpoint_plan(
    model: nn.Module,
    exp_cfg: DictConfig,
) -> list[str]:
    """Creates the FSDP activation checkpointing plan, i.e., the layer names of the individual FSDP units to checkpoint.

    Args:
        model: The model to apply activation checkpointing to.
        exp_cfg: The experiment configuration.

    Returns:
        A list of layer names to apply activation checkpointing.
    """

    activation_checkpoint_template = exp_cfg["fsdp"].get("activation_checkpoint_template", None)
    layers_to_grad_checkpoint = exp_cfg["fsdp"].get("fsdp_layers_to_activation_checkpoint", None)

    plan = []
    # use the explicit sharding template, e.g. model.transformer.blocks
    if activation_checkpoint_template:
        assert not layers_to_grad_checkpoint, "Use only one style of activation checkpoint."

        if not isinstance(activation_checkpoint_template, Iterable):
            activation_checkpoint_template = [activation_checkpoint_template]

        for template in activation_checkpoint_template:
            module = get_module(model, template)
            if hasattr(module, "__len__"):  # we have the name of a iterable layer, e.g. ModuleList
                n = len(module)
                plan.extend([f"{template}.{i}" for i in range(n)])
            else:  # directly have the name of a module
                plan.append(template)

    # infer from the class names the layers to shard e.g. Block
    elif layers_to_grad_checkpoint:
        # collect the layer names by matching their class names
        for name, module in model.named_modules():
            class_name = module.__class__.__name__
            # remove fsdp class name prefix
            class_name = (
                class_name[len(FSDP2_CLASS_NAME_PREFIX) :]
                if class_name.startswith(FSDP2_CLASS_NAME_PREFIX)
                else class_name
            )
            for layer in layers_to_grad_checkpoint:
                if class_name == layer:
                    plan.append(name)
                    break
    else:
        raise ValueError(
            "Cannot apply activation checkpoint, missing "
            "activation_checkpoint_template and fsdp_layers_to_activation_checkpoint."
        )

    return plan


def apply_ac_to_module(module: nn.Module, exp_cfg: DictConfig) -> nn.Module:
    """Apply activation checkpointing to a single module.

    Args:
        module: The module to apply activation checkpointing to.
        exp_cfg: The experiment configuration.

    Returns:
        The module with activation checkpointing applied.
    """

    _save_list = {
        torch.ops.aten.mm.default,
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
    }

    # 'full' and 'selective' result in different memory savings
    # 'full' will checkpoint all activations and 'selective' checkpoint the most expensive ones
    # this results in savings of around 70% of the 'full' checkpointing
    # while increasing computation time only 2.7% when compared with no checkpointing
    # https://huggingface.co/spaces/nanotron/ultrascale-playbook
    # we leave 'full' as default because sometimes 'selective' might not work
    ac_mode = exp_cfg["fsdp"].get("activation_checkpoint_mode", "full")
    mm_save_frequency = exp_cfg["fsdp"].get("selective_save_frequency", 2)

    if ac_mode == "full":
        return checkpoint_wrapper(module, preserve_rng_state=False)
    elif ac_mode == "selective":
        from torch.utils.checkpoint import (
            CheckpointPolicy,
            create_selective_checkpoint_contexts,
        )

        def _get_custom_policy(meta):
            def _custom_policy(ctx, func, *args, **kwargs):
                mode = "recompute" if ctx.is_recompute else "forward"
                mm_count_key = f"{mode}_mm_count"

                if func == torch.ops.aten.mm.default and func in _save_list:
                    meta[mm_count_key] += 1
                    to_save = (meta[mm_count_key] % mm_save_frequency) == 0
                elif func in _save_list:
                    to_save = True
                else:
                    to_save = False

                return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE

            return _custom_policy

        def selective_checkpointing_context_fn():
            meta = defaultdict(int)
            return create_selective_checkpoint_contexts(_get_custom_policy(meta))

        return checkpoint_wrapper(
            module,
            context_fn=selective_checkpointing_context_fn,
            preserve_rng_state=False,
        )
    else:
        raise ValueError(f"ac_mode needs to be 'full' or 'selective' ({ac_mode} was provided).")


def parallelize_model(
    model: nn.Module,
    device_mesh: DeviceMesh,
    exp_cfg: DictConfig,
    fsdp_grouping_plan: list[tuple[str, bool]] | None = None,
) -> nn.Module:
    """Parallelize model with FSDP2 and TP.

    Args:
        model: Model to be parallelized.
        device_mesh: Device mesh for parallelism.
        exp_cfg: Experiment configuration.
        fsdp_grouping_plan: Optional FSDP grouping plan.

    Returns:
        Parallelized model.
    """

    # how many ways to shard the model, typically number of gpus
    dp_shard = device_mesh["dp_shard"].size() if "dp_shard" in device_mesh.mesh_dim_names else 1
    # size of a TP unit, e.g. number of gpus that will act as a single device
    tp_size = device_mesh["tp_size"].size() if "tp_size" in device_mesh.mesh_dim_names else 1
    fsdp_type = exp_cfg["fsdp"]["sharding_strategy"].lower()

    if tp_size > 1:
        # bunch of requirements to do tp
        assert_can_do_tp(exp_cfg)
        tp_parallelize(model, device_mesh["tp"], exp_cfg)
        logger.info("Applied TP to the model.")

    if fsdp_type in ["no_shard", "full_shard", "shard_grad_op"]:
        if fsdp_type == "no_shard":
            # if we are not sharding, we need to replicate the model to all gpus
            assert dp_shard == 1, "dp_shard must be 1 for 'no_shard' and dp_replicate must be number of gpus."

        fsdp_config = {
            "mp_policy": (
                MixedPrecisionPolicy(
                    param_dtype=constants.generic.DTYPE_DICT[exp_cfg["fsdp"]["param_dtype"]],
                    reduce_dtype=constants.generic.DTYPE_DICT[exp_cfg["fsdp"]["reduce_dtype"]],
                    cast_forward_inputs=True,
                )
            ),
            "mesh": device_mesh["dp_replicate", "dp_shard"],
        }
        logger.info(f"FSDP2 config:\n{fsdp_config}")

        reshard_after_forward = fsdp_type != "shard_grad_op"
        for path in fsdp_grouping_plan:
            set_module(
                model,
                path,
                fully_shard(
                    get_module(model, path),
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                ),
            )
        model = fully_shard(model, **fsdp_config, reshard_after_forward=reshard_after_forward)

        logger.info(f"Sharded/replicated the model across devices.\n{model}")
    else:
        raise ValueError(
            f"invalid fsdp_type:  {fsdp_type}, only ['no_shard', 'full_shard', 'shard_grad_op'] are supported."
        )

    return model


def apply_ac(model: nn.Module, exp_cfg: DictConfig, fsdp_checkpointing_plan: list[str]) -> None:
    """Apply activation checkpointing to the model.

    Args:
        model: Model to apply activation checkpointing to.
        exp_cfg: Experiment configuration.
        fsdp_checkpointing_plan: List of module paths to apply checkpointing to.
    """

    for path in fsdp_checkpointing_plan:
        set_module(
            model,
            path,
            apply_ac_to_module(get_module(model, path), exp_cfg),
        )
    logger.info(f"Applied activation checkpoint with plan:\n{fsdp_checkpointing_plan}")


def init_fsdp2(
    model: nn.Module,
    device_mesh: DeviceMesh,
    exp_cfg: DictConfig,
    broadcast: bool = True,
) -> tuple[nn.Module, dict[Any, Any]]:
    """Apply FSDP2 sharding/TP to a model.

    Args:
        model: Model to be sharded.
        device_mesh: Device mesh for parallelism.
        exp_cfg: Experiment configuration.
        broadcast: Whether to broadcast parameters.

    Returns:
        Sharded model and FSDP extras.
    """

    # Extra FSDP specifics.
    extras = {}

    # sync model's parameters if not using the meta device init
    # this is needed when loading parts of the model from a checkpoint
    # all device will hold the complete model, which is then loaded into the main device and broadcasted
    if broadcast:
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        dist.barrier()

    # apply TP/FSDP2 sharding to the model
    model = parallelize_model(
        model,
        device_mesh,
        exp_cfg,
        fsdp_grouping_plan=build_grouping_plan(model, exp_cfg),
    )
    logger.info("Applied fsdp2 to the model")

    # apply activation checkpoint
    if exp_cfg["fsdp"].get("fsdp_activation_checkpointing", False):
        apply_ac(
            model,
            exp_cfg,
            fsdp_checkpointing_plan=build_activation_checkpoint_plan(model, exp_cfg),
        )

    if not broadcast:
        logger.info(
            "Did not broadcast parameters at fsdp2 init because no checkpoint is being loaded. "
            "This is the expected behavior if using the meta device init (the default)."
        )

    # Build the Context Parallelism plan.
    extras["context_parallel_plan"] = build_cp_plan(exp_cfg)

    return model, extras
