# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import logging
from collections.abc import Callable, Iterable
from typing import Any, Literal

import torch
import torch.nn as nn
from timm.layers import to_2tuple
from timm.layers.weight_init import trunc_normal_, variance_scaling_

from l3m.constants.typing import DATA_DICT
from l3m.helpers.dist.utils import replicate_tensor_if_distributed, tensor_to_dtensor
from l3m.model.meta_models import ReadWriteBlock
from l3m.model.preprocessors.pos_embed import sinusoidal

__all__ = [
    "ViTPreprocessor",
    "ViTPreprocessorWithRegisters",
    "OpenAICLIPViTProcessor",
    "OpenAICLIPViTProcessorWithRegisters",
    "PatchEmbed",
    "PatchEmbedLinear",
    "TokenSubSampler",
    "TextPreprocessor",
]

logger = logging.getLogger("l3m")


class ViTPreprocessor(ReadWriteBlock):
    """Vision Transformer (ViT) preprocessor that patchifies images and adds positional embeddings.

    It takes an image as input, divides it into patches using a patchifier,
    and adds positional embeddings to the resulting tokens. It also supports
    dropping patches based on a mask and prepending a class token.

    Args:
        patchifier: Module that converts images into patches/tokens.
        pos_embed_type: Type of positional embedding to use.
        drop_patches: Whether to drop patches based on a mask.
        cls_token: Whether to prepend a class token.
        attn_mask_read_key: Key for the attention mask in the data dictionary.
    """

    def __init__(
        self,
        patchifier: nn.Module,
        pos_embed_type: Literal["absolute", "sincos"] | None = None,
        drop_patches: bool = False,
        cls_token: bool = True,
        attn_mask_read_key: str = "mask",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.patchifier = patchifier
        self.cls_token = nn.Parameter(torch.empty(1, 1, self.patchifier.embed_dim)) if cls_token else None

        self.pos_embed = None
        if pos_embed_type in ["absolute", "sincos"]:
            self.pos_embed = nn.Parameter(
                torch.empty(
                    1,
                    self.patchifier.num_patches + (self.cls_token is not None),
                    self.patchifier.embed_dim,
                ),
                requires_grad=(pos_embed_type == "absolute"),
            )

        self.pos_embed_type = pos_embed_type
        self.drop_patches = drop_patches
        self.attn_mask_read_key = attn_mask_read_key

        self.init_weights()

    def init_weights(self) -> None:
        if self.pos_embed_type == "absolute":
            torch.nn.init.normal_(self.pos_embed, std=0.02)
        elif self.pos_embed_type == "sincos":
            assert 1 <= len(self.patchifier.grid_size) <= 3, (
                "grid_size has to contain 1~3 values for sincos positional embeddings."
            )
            sincos_init = None
            if len(self.patchifier.grid_size) <= 2:
                sincos_init = sinusoidal.get_2d_sincos_pos_embed(
                    self.pos_embed.shape[-1],
                    self.patchifier.grid_size,
                    cls_token=(self.cls_token is not None),
                )
            elif len(self.patchifier.grid_size) == 3:
                sincos_init = sinusoidal.get_3d_sincos_pos_embed(
                    self.pos_embed.shape[-1],
                    self.patchifier.grid_size,
                    cls_token=(self.cls_token is not None),
                )
            else:
                raise ValueError(f"Unsupported grid size {self.patchifier.grid_size}")

            # create tensor and convert it to dtensor if needed
            sincos_init = tensor_to_dtensor(torch.from_numpy(sincos_init).float().unsqueeze(0), self.pos_embed)
            self.pos_embed.data.copy_(sincos_init)

        elif self.pos_embed_type is not None:
            raise ValueError(f"Unsupported positional embedding {self.pos_embed_type}.")

        if self.cls_token is not None:
            torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d) following MAE
        if hasattr(self.patchifier, "proj"):
            w = self.patchifier.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        image, mask = (
            data_dict[self.read_key],
            data_dict.get(self.attn_mask_read_key, None),
        )
        tokens = self.patchifier(image)
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls_token, tokens], dim=1)

        B, N, D = tokens.shape
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed[:, :N]

        if self.drop_patches and mask is not None:
            if self.cls_token is not None:
                cls_token, tokens = tokens[:, 0:1], tokens[:, 1:]
            tokens = tokens[~mask].reshape(B, -1, D)
            if self.cls_token is not None:
                tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = tokens.contiguous() if not tokens.is_contiguous() else tokens  # megablocks uses .view()
        data_dict[self.write_key] = tokens
        return data_dict


class ViTPreprocessorWithRegisters(ViTPreprocessor):
    """Vision Transformer (ViT) preprocessor that appends learnable registers to the patchified image tokens.

    Extends :class:`ViTPreprocessor` to add learnable register tokens, which can be appended either to the left or right
    of the original tokens.

    Args:
        num_registers: Number of registers to append.
        append_side: Side to append registers to.
        registers_mask_write_key: Key to write the registers mask in the data dictionary.
    """

    def __init__(
        self,
        *args: Any,
        num_registers: int,
        append_side: Literal["right", "left"] = "right",
        registers_mask_write_key: str = "registers_mask",
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        assert append_side in [
            "right",
            "left",
        ], f"append_side:`{append_side}` is not supported. Please pass `right` or `left`."
        self.append_side = append_side
        self.num_registers = num_registers
        self.registers_mask_write_key = registers_mask_write_key
        self.registers = nn.Parameter(torch.empty(1, num_registers, self.patchifier.embed_dim))

        self.init_weights()

    def init_weights(self) -> None:
        super().init_weights()

        torch.nn.init.normal_(self.registers, std=0.02)

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        data_dict = super().forward(data_dict)

        # Overwrite the tokens with `self.write_key`
        tokens = data_dict[self.write_key]
        B, N, D = tokens.shape
        registers = self.registers.expand(B, -1, -1)
        if self.append_side == "right":
            tokens = torch.cat((tokens, registers), dim=1)
            registers_mask = torch.zeros_like(tokens, device=tokens.device)
            registers_mask[:, N:] = 1
        else:
            tokens = torch.cat((registers, tokens), dim=1)
            registers_mask = torch.zeros_like(tokens, device=tokens.device)
            registers_mask[:, : self.num_registers] = 1

        data_dict[self.write_key] = tokens
        data_dict[self.registers_mask_write_key] = registers_mask.bool()
        return data_dict


class OpenAICLIPViTProcessor(ViTPreprocessor):
    """Vision Transformer (ViT) preprocessor compatible with OpenAI CLIP checkpoints.

    This preprocessor adjusts the order of operations to match the specific requirements of OpenAI CLIP models.
    The order of operations is as follows:

    1. patch embedding
    2. add cls_tokens
    3. add positional embedding
    4. apply the layernorm
    """

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        image = data_dict[self.read_key]
        tokens = self.patchifier.proj(image).flatten(2).transpose(1, 2)
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls_token, tokens], dim=1)
        B, N, D = tokens.shape
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed[:, :N]
        tokens = self.patchifier.norm(tokens)
        data_dict[self.write_key] = tokens
        return data_dict


class OpenAICLIPViTProcessorWithRegisters(OpenAICLIPViTProcessor):
    """Vision Transformer (ViT) preprocessor compatible with OpenAI CLIP checkpoints, with learnable registers.

    Extends OpenAICLIPViTProcessor to add learnable register tokens, which can be appended either to the left or right
    of the original tokens. This combines the OpenAI CLIP compatibility with the register token functionality.

    Args:
        num_registers: Number of registers to append.
        append_side: Side to append registers to.
        registers_mask_write_key: Key for the registers mask in the data dictionary.
    """

    def __init__(
        self,
        *args: Any,
        num_registers: int,
        append_side: Literal["right", "left"] = "right",
        registers_mask_write_key: str = "registers_mask",
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        assert append_side in [
            "right",
            "left",
        ], f"append_side:`{append_side}` is not supported. Please pass `right` or `left`."
        self.append_side = append_side
        self.num_registers = num_registers
        self.registers_mask_write_key = registers_mask_write_key
        self.registers = nn.Parameter(torch.randn(1, num_registers, self.patchifier.embed_dim) * 0.02)

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        data_dict = super().forward(data_dict)

        # Overwrite the tokens with `self.write_key`
        tokens = data_dict[self.write_key]
        B, N, D = tokens.shape
        registers = self.registers.expand(B, -1, -1)
        if self.append_side == "right":
            tokens = torch.cat((tokens, registers), dim=1)
            registers_mask = torch.zeros_like(tokens, device=tokens.device)
            registers_mask[:, N:] = 1
        else:
            tokens = torch.cat((registers, tokens), dim=1)
            registers_mask = torch.zeros_like(tokens, device=tokens.device)
            registers_mask[:, : self.num_registers] = 1

        data_dict[self.write_key] = tokens
        data_dict[self.registers_mask_write_key] = registers_mask.bool()
        return data_dict


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding.

    This module converts a 2D image into a sequence of patch embeddings.  It divides the input image into
    non-overlapping patches, projects each patch into a higher-dimensional embedding space, and applies
    a normalization layer.

    Args:
        img_size: The input image size (assumed to be square).
        patch_size: The size of each patch (assumed to be square).
        in_chans: The number of input channels in the image.
        embed_dim: The dimension of the patch embedding space.
        norm_layer: Normalization layer to apply to the embeddings.
        use_bias: Whether to use bias in the Conv2d projection layer.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable[[int], nn.Module] = None,
        use_bias: bool = True,
    ):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size, self.embed_dim = img_size, embed_dim
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=use_bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class PatchEmbedLinear(nn.Module):
    """2D Image to Patch Embedding using a Linear Projection.

    This module converts a 2D image into a sequence of patch embeddings using a linear projection.
    It divides the input image into non-overlapping patches using `unfold`, then projects each flattened patch
    into a higher-dimensional embedding space using a linear layer, and applies a normalization layer.

    Args:
        img_size: The input image size (assumed to be square).
        patch_size: The size of each patch (assumed to be square).
        in_chans: The number of input channels in the image.
        embed_dim: The dimension of the patch embedding space.
        norm_layer: Normalization layer to apply to the embeddings.
        weight_init_style: Weight initialization style.
        init_std: Standard deviation for weight initialization (trunc_normal only).
        use_bias: Whether to use bias in the Linear projection layer.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable[[int], nn.Module] | None = None,
        use_bias: bool = True,
        weight_init_style: Literal["trunc_normal", "kaiming", "fan_in", "pytorch"] = "pytorch",
        init_std: float = 0.02,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size, self.embed_dim = img_size, embed_dim
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Linear(patch_size[0] * patch_size[1] * in_chans, embed_dim, bias=use_bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.weight_init_style = weight_init_style
        self.init_std = init_std

        self.init_weights()

    def init_weights(self) -> None:
        if self.weight_init_style == "trunc_normal":
            trunc_normal_(self.proj.weight, std=self.init_std)
        elif self.weight_init_style == "kaiming":
            nn.init.kaiming_normal_(self.proj.weight, mode="fan_in", nonlinearity="linear")
        elif self.weight_init_style == "fan_in":
            variance_scaling_(self.proj.weight, scale=1.0, mode="fan_in", distribution="normal")
        elif self.weight_init_style == "pytorch":
            self.proj.reset_parameters()

        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        x = self.proj(x)
        x = self.norm(x)
        return x


class TokenSubSampler:
    """Token SubSampler.

    This class subsamples a sequence of tokens by selecting a slice within a specified range.

    Args:
        subsample_range: A tuple containing the start and end indices (exclusive) of the desired slice.
    """

    def __init__(self, subsample_range: tuple[int, int]):
        self.start, self.end = subsample_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.start : self.end]


class TextPreprocessor(ReadWriteBlock):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        context_length: int = 77,
        cls_token: bool = True,
        pos_embed_type: Literal["absolute", "sincos"] = "absolute",
        inference_read_key: str = "prompt",
        init_style: (Literal["normal", "sparse_transformer", "uniform"] | None) = "normal",
        init_std: float = 0.02,
        norm_layer: Callable[[int], nn.Module] | None = None,
        cls_token_init: Literal["normal", "zeros"] = "normal",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.text_embedding = nn.Embedding(vocab_size, embed_dim)

        self.positional_embedding = None
        if pos_embed_type in ["absolute", "sincos"]:
            self.positional_embedding = nn.Parameter(
                torch.empty(context_length + int(cls_token), embed_dim),
                requires_grad=(pos_embed_type == "absolute"),
            )
        self.pos_embed_type = pos_embed_type
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.init_style = init_style
        self.init_std = init_std
        self.context_length = context_length

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim)) if cls_token else None
        self.vocab_size = vocab_size
        self.inference_read_key = inference_read_key
        self.cls_token_init = cls_token_init

        self.init_weights()

    def init_weights(self) -> None:
        if self.pos_embed_type == "absolute":
            torch.nn.init.normal_(self.positional_embedding, std=0.01)
        elif self.pos_embed_type == "sincos":
            sincos_init = sinusoidal.get_2d_sincos_pos_embed(
                self.positional_embedding.shape[-1],
                self.context_length,
                cls_token=(self.cls_token is not None),
            )
            # create tensor and convert it to dtensor if needed
            sincos_init = tensor_to_dtensor(torch.from_numpy(sincos_init).float(), self.positional_embedding)
            self.positional_embedding.data.copy_(sincos_init)
        elif self.pos_embed_type is not None:
            raise ValueError(f"Unsupported positional embedding {self.pos_embed_type}.")

        # cls token
        if self.cls_token is not None:
            if self.cls_token_init == "normal":
                torch.nn.init.normal_(self.cls_token, std=0.02)
            elif self.cls_token_init == "zeros":
                self.cls_token.data.fill_(0.0)
            else:
                raise ValueError(f"cls_token_init={self.cls_token_init} not supported, use 'normal' or 'zeros'")

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Embedding):
            if self.init_style == "normal":
                torch.nn.init.normal_(m.weight, mean=0.0, std=self.init_std)
            elif self.init_style == "sparse_transformer":
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.125 * (self.embed_dim**-0.5))
            elif self.init_style == "uniform":
                torch.nn.init.uniform_(m.weight, -0.1, 0.1)
            elif self.init_style is None:
                pass
            else:
                raise ValueError(f"Invalid initialization style {self.init_style}.")

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        if not self.generation_mode:
            tokens = data_dict[self.read_key]
        else:
            tokens = data_dict.get(self.inference_read_key, data_dict[self.read_key])

        tokens = self.text_embedding(tokens)
        if self.cls_token is not None:
            # we make sure everything is a dtensor, it might be the case that the
            # cls token is not converted and there will be a mismatch in types
            cls_token = replicate_tensor_if_distributed(self.cls_token, tokens)
            cls_token = cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls_token, tokens], dim=1)

        if self.positional_embedding is not None:
            _, seq_length, _ = tokens.shape
            # we make sure everything is a dtensor, it might be the case that the
            # pos embed is not converted and there will be a mismatch in types
            pos_embed = replicate_tensor_if_distributed(self.positional_embedding, tokens)
            max_length = pos_embed.shape[0]
            max_seq_length = min(seq_length, max_length)
            if max_seq_length < seq_length and not self.training:
                logger.info(f"Text {seq_length} is larger than model context length {max_length}.")

            tokens = tokens[:, :max_seq_length, :] + torch.unsqueeze(pos_embed, 0)[:, :max_seq_length, :]
        data_dict[self.write_key] = tokens
        return data_dict


def reshape(x: torch.Tensor, shape: Iterable[int]) -> torch.Tensor:
    return torch.reshape(x, tuple(shape))
