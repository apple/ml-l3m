# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import torch.nn as nn
from PIL import Image

__all__ = ["expand_to_square", "PadToSquare"]


def expand_to_square(
    pil_img: Image.Image,
    background_color: float | tuple[float, ...] | str | None,
) -> Image.Image:
    """Expands a PIL image to a square, padding with a background color.

    Args:
        pil_img: The input PIL image.
        background_color: The background color to use for padding.

    Returns:
        The expanded square PIL image.
    """

    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class PadToSquare(nn.Module):
    """Pads an image to a square shape.

    Args:
        mean: Mean pixel values for padding.
    """

    def __init__(self, mean: tuple[float, ...] = (0.485, 0.456, 0.406)):
        super().__init__()

        self.mean = mean

    def forward(self, image: Image.Image) -> Image.Image:
        image = expand_to_square(image, tuple(int(x * 255) for x in self.mean))
        return image

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean})"
