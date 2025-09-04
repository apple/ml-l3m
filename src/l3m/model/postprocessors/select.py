# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Any

import torch

from l3m.constants.typing import DATA_DICT
from l3m.model.meta_models import ReadWriteBlock

__all__ = ["SelectTokens", "SelectTokensWithMask", "ExtractCLS", "ExtractEOS"]


class SelectTokens(ReadWriteBlock):
    """Selects a range of tokens from a tensor along a specified dimension.

    Args:
        start: Starting index for token selection.
        end: Ending index for token selection. Defaults to :obj:`None`, which selects until the end of the dimension.
        dim: Dimension along which to select tokens.
    """

    def __init__(self, start: int, end: int | None = None, dim: int = 1, **kwargs: Any):
        super().__init__(**kwargs)

        self.start, self.end = start, end
        self.dim = dim

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        tokens = data_dict[self.read_key]
        start = self.start
        end = self.end or tokens.size(self.dim)
        tokens = torch.index_select(
            tokens,
            dim=self.dim,
            index=torch.arange(start, end).to(tokens.device),
        )
        data_dict[self.write_key] = tokens

        return data_dict


class SelectTokensWithMask(ReadWriteBlock):
    """Selects tokens based on a mask and writes them to a new key.

    Args:
        mask_read_key: Key in the data_dict containing the mask.
        inverted_write_key: Key to write the non-selected tokens to.
    """

    def __init__(self, mask_read_key: str, inverted_write_key: str | None = None, **kwargs: Any):
        super().__init__(**kwargs)

        self.mask_read_key = mask_read_key
        self.inverted_write_key = inverted_write_key

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        tokens = data_dict[self.read_key]
        mask = data_dict[self.mask_read_key]
        assert tokens.ndim == 3, f"tokens of shape ({tokens.shape}) needs to be of shape [b, s, d]"
        B, _, D = tokens.shape
        selected_tokens = tokens[mask].reshape(B, -1, D)

        data_dict[self.write_key] = selected_tokens
        if self.inverted_write_key is not None:
            non_selected_tokens = tokens[~mask].reshape(B, -1, D)
            data_dict[self.inverted_write_key] = non_selected_tokens

        return data_dict


class ExtractCLS(ReadWriteBlock):
    """Extracts the CLS token from a sequence of embeddings.

    Args:
        index: CLS token index.
        dim: Sequence dimension.
        keepdim: Whether to keep the sequence dim.
    """

    def __init__(
        self,
        index: int = 0,
        dim: int = 1,
        keepdim: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.index = index
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        x = data_dict[self.read_key]
        cls_token = torch.select(x, self.dim, self.index)
        if self.keepdim:
            cls_token = cls_token.unsqueeze(1)
        data_dict[self.write_key] = cls_token

        return data_dict


class ExtractEOS(ReadWriteBlock):
    """Extracts the EOS token embedding from a sequence of embeddings.

    Args:
        eos_token_id: The ID of the EOS token.
        keepdim: Whether to keep the dimension of the extracted token.
        text_read_key: Key to read the text from the data dictionary.
    """

    def __init__(
        self,
        eos_token_id: int = 0,
        keepdim: bool = False,
        text_read_key: str = "text",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.keepdim = keepdim
        self.eos_token_id = eos_token_id
        self.text_read_key = text_read_key

    def forward(self, data_dict: DATA_DICT, **_: Any) -> DATA_DICT:
        x = data_dict[self.read_key]

        input_ids = data_dict[self.text_read_key]

        # create the token mask
        token_mask = input_ids == self.eos_token_id  # shape: (b, seq_len)

        # find the index of the EOS token for each sequence
        eos_token_index = torch.argmax(token_mask.float(), dim=1)  # shape: (b,)

        # expand dimensions to match the input tensor x
        eos_token_index = eos_token_index.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, x.size(-1))  # shape: (b, 1, dim)

        # gather the EOS token embeddings
        eos_token = torch.gather(x, 1, eos_token_index)  # shape: (b, 1, dim)

        if not self.keepdim:
            eos_token = eos_token.squeeze()

        data_dict[self.write_key] = eos_token
        return data_dict
