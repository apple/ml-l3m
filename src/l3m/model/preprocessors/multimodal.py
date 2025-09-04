# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import logging
from typing import Any, Literal

import torch
import torch.nn as nn

from l3m.constants.typing import DATA_DICT
from l3m.helpers.dist.utils import replicate_tensor_if_distributed, tensor_to_dtensor
from l3m.model.meta_models import ReadWriteBlock
from l3m.model.preprocessors.pos_embed import sinusoidal

__all__ = ["VisionAndTextPreprocessor"]

TEXT_VISION_CONCAT = "text||vision"
VISION_TEXT_CONCAT = "vision||text"
REPLACE_IMAGE_TOKEN = "replace_image_token"
TEXT_ONLY = "text"
MERGE_PROTOCOLS = (
    TEXT_VISION_CONCAT,
    VISION_TEXT_CONCAT,
    REPLACE_IMAGE_TOKEN,
    TEXT_ONLY,
)


logger = logging.getLogger("l3m")


class VisionAndTextPreprocessor(ReadWriteBlock):
    """Combines vision and text inputs into a single sequence of tokens.

    This module preprocesses vision and text inputs separately using dedicated
    preprocessors, and then merges them into a single sequence based on the
    specified merge protocol. It also supports positional embeddings.

    Args:
        vision_read_key: Key in the data dictionary for the vision input.
        text_read_key: Key in the data dictionary for the text input.
        vision_preprocessor: Preprocessor module for the vision input.
        text_preprocessor: Preprocessor module for the text input.
        write_key: Key in the data dictionary to write the merged tokens to.
        merge_protocol: Protocol for merging vision and text tokens.
        padding_side: Side to pad the text tokens.
        ignore_index: Index to ignore during loss calculation.
        context_length: Maximum sequence length.
        pos_embed_type: Type of positional embedding to use.
        embed_dim: Embedding dimension.
    """

    def __init__(
        self,
        vision_read_key: str,
        text_read_key: str,
        vision_preprocessor: nn.Module,
        text_preprocessor: nn.Module,
        write_key: str,
        merge_protocol: Literal[
            "text||vision",
            "vision||text",
            "replace_image_token",
            "text",
        ] = "vision||text",
        padding_side: str = "left",
        ignore_index: int = 0,
        context_length: int = 334,
        pos_embed_type: Literal["absolute", "sincos"] | None = None,
        embed_dim: int = 1024,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.vision_read_key, self.text_read_key = vision_read_key, text_read_key
        self.write_key = write_key
        assert merge_protocol in MERGE_PROTOCOLS, (
            f"{merge_protocol} is not supported. Please pass one of {MERGE_PROTOCOLS}."
        )
        self.merge_protocol = merge_protocol

        self.vision_preprocessor = vision_preprocessor
        self.text_preprocessor = text_preprocessor

        self.cls_token_id = -200

        self.context_length = context_length
        self.padding_side = padding_side
        self.ignore_index = ignore_index

        self.positional_embedding = None
        self.pos_embed_type = pos_embed_type
        if pos_embed_type in ["absolute", "sincos"]:
            self.positional_embedding = nn.Parameter(
                torch.empty(context_length, embed_dim),
                requires_grad=(pos_embed_type == "absolute"),
            )

        self.init_weights()

    def init_weights(self) -> None:
        # Init following CLIP
        if self.pos_embed_type == "absolute":
            torch.nn.init.normal_(self.positional_embedding, std=0.01)
        elif self.pos_embed_type == "sincos":
            sincos_init = sinusoidal.get_2d_sincos_pos_embed(
                self.positional_embedding.shape[-1],
                (self.context_length, 1),
            )
            # create tensor and convert it to dtensor if needed
            sincos_init = tensor_to_dtensor(
                torch.from_numpy(sincos_init).float().unsqueeze(0),
                self.positional_embedding,
            )
            self.positional_embedding.data.copy_(sincos_init)
        elif self.pos_embed_type is not None:
            raise ValueError(f"Unsupported positional embedding {self.pos_embed_type}.")

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        """Pre-processes the vision+text inputs.

        Args:
            data_dict: Input data containing the following keys:

                - ``'{self.vision_read_key}'``
                - ``'{self.text_read_key}'``

        Returns:
            Output that contains:

            - ``'{self.write_key}'`` - multimodal tokens of shape ``[B, L, D]``.
        """
        if self.generation_mode:
            assert self.vision_preprocessor.read_key != self.vision_preprocessor.write_key, (
                "In generation mode, the read and write key in the vision_preprocessor "
                "can not be the same, otherwise they will be overwritten when generating tokens!"
            )

        data_dict = self.vision_preprocessor(data_dict)
        # this must be after to override the vision mask
        data_dict = self.text_preprocessor(data_dict)

        # Make sure the vision and text key exists
        assert self.vision_read_key in data_dict, data_dict.key()
        assert self.text_read_key in data_dict, data_dict.key()

        vision_tokens = data_dict[self.vision_read_key]
        text_tokens = data_dict[self.text_read_key]
        assert vision_tokens.size(-1) == text_tokens.size(-1), (
            f"Vision and text tokens must have the same embedding dimension \
            (i.e. dim=-1), got {vision_tokens.size(-1)} and {text_tokens.size(-1)}."
        )

        if self.generation_mode:
            assert self.vision_preprocessor.read_key != self.vision_preprocessor.write_key, (
                "In generation mode, the read and write key in the vision_preprocessor can not be the same!"
            )

        if not self.generation_mode:
            input_ids = data_dict[self.text_preprocessor.read_key]
            target_ids = input_ids if "target_ids" not in data_dict else data_dict["target_ids"]
        else:
            input_ids = data_dict.get(
                self.text_preprocessor.inference_read_key,
                data_dict[self.text_preprocessor.read_key],
            )
            target_ids = input_ids

        target_ids = target_ids.clone()

        if target_ids.shape[1] == text_tokens.shape[1] - 1:  # cls token, -> target_ids.shape[1] == text_tokens.shape[1]
            target_ids = torch.cat(
                (torch.ones_like(target_ids[:, :1]) * (self.cls_token_id), target_ids),
                dim=1,
            )

        if self.merge_protocol == VISION_TEXT_CONCAT:
            tokens = torch.cat((vision_tokens, text_tokens), dim=1)
        elif self.merge_protocol == REPLACE_IMAGE_TOKEN:
            assert "image_token_id" in data_dict, (
                f"{self.merge_protocol} is selected but image_token_id is not provided in {data_dict.keys()}"
            )

            image_token_id = data_dict["image_token_id"][0]

            if input_ids.shape[1] == text_tokens.shape[1] - 1:  #  cls token
                input_ids = torch.cat(
                    (torch.ones_like(input_ids[:, :1]) * self.cls_token_id, input_ids),
                    dim=1,
                )

            # For text-only sequences, the `vision_tokens` include dummy image patches
            # therefore, we need to filter them out before replacing the placeholders
            # `samples_with_images_idx` will include ids of samples that include images
            if "nonpad_images_mask" in data_dict:
                nonpad_images_mask = data_dict["nonpad_images_mask"]
            else:
                nonpad_images_mask = torch.where(torch.any(input_ids == image_token_id, dim=-1))

            # Replace all <image> placeholder tokens with their corresponding
            # vision features
            batch_idx, token_idx = torch.where(input_ids == image_token_id)
            nonpadded_vision_tokens = vision_tokens.to(text_tokens.dtype)[nonpad_images_mask].flatten(0, 1)

            try:
                text_tokens[batch_idx, token_idx] = nonpadded_vision_tokens
            except Exception as e:  # noqa: BLE001
                min_size = min(batch_idx.shape[0], nonpadded_vision_tokens.shape[0])
                text_tokens[batch_idx[:min_size], token_idx[:min_size]] = nonpadded_vision_tokens[:min_size]
                logger.warning(
                    f"Error in populating the special image tokens due to {e}.\n"
                    f"nonpadded_vision_tokens: {nonpadded_vision_tokens.shape}, "
                    f"text_tokens: {text_tokens.shape}, image_token_id: {image_token_id} input_ids: {input_ids}"
                )

            tokens = text_tokens
        else:
            raise NotImplementedError(f"self.merge_protocol: {self.merge_protocol} not implemented")

        if self.positional_embedding is not None:
            # we make sure everything is a dtensor, it might be the case that the
            # pos embed is not converted and there will be a mismatch in types
            pos_embed = replicate_tensor_if_distributed(self.positional_embedding, tokens)
            _, seq_length, _ = tokens.shape
            tokens = tokens[:, :seq_length, :] + torch.unsqueeze(pos_embed, 0)[:, :seq_length, :]

        data_dict[self.write_key] = tokens
        return data_dict
