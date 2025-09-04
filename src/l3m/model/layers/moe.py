# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import torch
import torch.nn as nn

try:
    from megablocks.layers.arguments import Arguments as megablocks_arguments
    from megablocks.layers.dmoe import dMoE as megablocks_dmoe
    from megablocks.layers.moe import MoE as megablocks_moe
except ImportError as e:
    import logging

    logger = logging.getLogger("l3m")
    logger.warning(f"Could not import `moe`. MoE training is not possible: {e}")

    megablocks_moe = None
    megablocks_dmoe = None

__all__ = ["MoE"]


class MoE(nn.Module):
    MOE_CLASS = {
        "moe": megablocks_moe,
        "dmoe": megablocks_dmoe,
    }

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        use_bias: bool = True,
        return_bias: bool = False,
        num_experts: int = 1,
        top_k: int = 1,
        normalize_expert_weights: float = 0.0,
        uniform_expert_assignment: bool = False,
        load_balancing_loss_weight: float = 0.1,
        capacity_factor: int = 1,
        router_jitter_eps: None | float = None,
        mlp_type: Literal["mlp", "glu"] = "glu",
        mlp_impl: Literal["sparse", "grouped"] = "grouped",
        memory_optimized_mlp: bool = False,
        moe_class: Literal["moe", "dmoe"] = "dmoe",
        multiple_of: int = 256,
        dtype: torch.dtype = torch.float32,
        num_layers: int = 1,
        load_balancing_in_fp32: bool = False,
        pipeline_model_parallel_size: int = 1,
        moe_expert_model_parallelism: bool = False,
        expert_parallel_group: None | torch.distributed.ProcessGroup = None,
        num_layers_per_virtual_pipeline_stage: None | int = None,
        device: torch.device | None = None,
        **_: Any,
    ):
        """Wrapper around the megablocks {MoE, dMoE} layers

        Args:
            in_features: input embedding size
            hidden_features: hidden layer size
            act_layer: activation function to use
            use_bias: add bias to final moe module output
            return_bias: return bias along with moe output
            num_experts: number of experts to use
            top_k: number of experts to route per token
            normalize_expert_weights: p in ell_p norm used to normalize top-k expert weights (note: this is not softmax)
            uniform_expert_assignment: whether to use uniform expert assignment or not (for testing, default is False)
            load_balancing_loss_weight: weighting factor for the load balancing loss
            capacity_factor: capacity factor for the moe layer (not used for dMoE)
            router_jitter_eps: multiplicative (1-eps, 1+eps) noise to router logits if not None
            mlp_type: mlp or glu (defaults to glu)
            mlp_impl: uses grouped GEMM for dmoe computation iff 'grouped'
            memory_optimized_mlp: whether to use memory optimized mlp or not
            moe_class: Which MoE type to use: moe | dmoe
            multiple_of: ensures that the hidden layer size is a multiple of this value
            dtype: float32 by default (this is changed at model level after instantiating)
            num_layers: number of MoE layers in trunk
            load_balancing_in_fp32: whether to compute the load balancing loss in fp32 or not
            pipeline_model_parallel_size: size for model parallelism (defaults to 1)
            moe_weight_parallelism: enable weight parallelism for MoE layer
            moe_expert_model_parallelism: enable model parallelism for MoE experts
            expert_parallel_group: defaults to None
            weight_parallel_group: defaults to None
            num_layers_per_virtual_pipeline_stage: overrides num_layers for lbl computation (defaults to None/disabled)
            device: cpu by default (this is changed at model level after instantiating)
        """
        # setup
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        assert top_k <= num_experts, "top_k should be less than or equal to num_experts"
        valid_classes = ", ".join(list(self.MOE_CLASS.keys()))
        assert moe_class in self.MOE_CLASS, f"{moe_class} is not supported. Available modules: {valid_classes}"

        # adjust hidden features to be multiple of `multiple_of` (taken from ffn.py)
        hidden_features = int(2 * hidden_features / 3)
        hidden_features = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)

        # setup arguments (input) for the megablocks moe layer
        self.megablocks_arguments = megablocks_arguments(
            hidden_size=in_features,
            ffn_hidden_size=hidden_features,
            activation_fn=act_layer(),
            bias=use_bias,
            bf16=dtype == torch.bfloat16,
            fp16=dtype == torch.float16,
            device=device,
            moe_num_experts=num_experts,
            moe_capacity_factor=capacity_factor,
            moe_top_k=top_k,
            moe_normalize_expert_weights=normalize_expert_weights,
            moe_loss_weight=load_balancing_loss_weight,
            moe_jitter_eps=router_jitter_eps,
            mlp_impl=mlp_impl,
            mlp_type=mlp_type,
            return_bias=return_bias,
            num_layers=num_layers,
            uniform_expert_assignment=uniform_expert_assignment,
            moe_lbl_in_fp32=load_balancing_in_fp32,
            memory_optimized_mlp=memory_optimized_mlp,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            moe_expert_model_parallelism=moe_expert_model_parallelism,
            expert_parallel_group=expert_parallel_group,
            num_layers_per_virtual_pipeline_stage=num_layers_per_virtual_pipeline_stage,
        )

        # create the MoE or dMoE layer based on the `use_dropless` flag
        self.moe = self.MOE_CLASS[moe_class](self.megablocks_arguments)
        self.init_weights()

    def init_weights(self) -> None:
        init_method = partial(torch.nn.init.normal_, mean=0.0, std=0.02)  # from megablocks
        with torch.no_grad():
            init_method(self.moe.experts.mlp.w1)
            init_method(self.moe.experts.mlp.w2)
            init_method(self.moe.experts.mlp.v1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.moe(x)
        return out
