# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Any, NamedTuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

__all__ = ["HFWrapper"]


class HFWrapper(PreTrainedModel):
    """Wraps a model for use with Hugging Face's Transformers library.

    Args:
        cfg: Configuration for the model.
    """

    def __init__(self, cfg: DictConfig):
        config = PretrainedConfig(
            vocab_size=cfg.constants.vocab_size,
            is_encoder_decoder=False,
        )
        super().__init__(config)
        self.model = instantiate(cfg.model)["meta_model"]
        self.input_key = cfg.model.meta_model.preprocessor.read_key
        self.output_key = cfg.model.meta_model.head.write_key

    def forward(self, input_ids: torch.Tensor, **_: Any) -> CausalLMOutput:
        output = self.model({self.input_key: input_ids})
        return CausalLMOutput(logits=output[self.output_key])

    def load_pretrained(self, state_dict: dict[str, torch.Tensor]) -> NamedTuple:
        status = self.model.load_state_dict(state_dict)
        return status

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor | None]:
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        return model_inputs
