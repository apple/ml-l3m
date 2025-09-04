# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import gzip
import html
import os
from functools import lru_cache
from typing import Any, Literal

import ftfy
import regex
import torch
from tokenizers.normalizers import NFD, Lowercase, Sequence, StripAccents

__all__ = ["SimpleTokenizer", "HFTokenizer"]


IMAGE_TOKEN = "<image>"
END_OF_UTTERANCE = "<eou>"


def basic_clean(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    text = regex.sub(r"\s+", " ", text)
    text = text.strip()
    return text


@lru_cache
def default_bpe() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache
def bytes_to_unicode() -> dict[str, str]:
    """Returns dict of utf-8 byte and the corresponding unicode string.

    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs, strict=True))


def get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class SimpleTokenizer:
    def __init__(
        self,
        bpe_path: str,
        special_tokens: list[str] | None = None,
        context_length: int = 77,
        pad_token: int | str = 0,
        add_eos_token: bool = True,
        add_bos_token: bool = True,
        normalize: bool = False,
    ):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        if not special_tokens:
            special_tokens = ["<start_of_text>", "<end_of_text>"]
        else:
            special_tokens = ["<start_of_text>", "<end_of_text>"] + special_tokens
        vocab.extend(special_tokens)
        self.encoder = dict(zip(vocab, range(len(vocab)), strict=True))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges)), strict=True))
        self.cache = {t: t for t in special_tokens}
        special = "|".join(special_tokens)
        self.pat = regex.compile(
            special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            regex.IGNORECASE,
        )

        self.vocab_size = len(self.encoder)
        self.all_special_ids = [self.encoder[t] for t in special_tokens]
        self.context_length = context_length

        self.eos_token = "<end_of_text>"
        self.bos_token = "<start_of_text>"
        self.eos_token_id = self.encoder["<end_of_text>"]
        self.bos_token_id = self.encoder["<start_of_text>"]

        if isinstance(pad_token, int):
            self.pad_token_id = pad_token
            self.pad_token = self.decoder[pad_token]
        elif isinstance(pad_token, str):
            assert pad_token in special_tokens
            self.pad_token_id = self.encoder[pad_token]
            self.pad_token = pad_token
        else:
            raise NotImplementedError(type(pad_token))

        self.add_eos_token = add_eos_token
        self.add_bos_token = add_bos_token

        self.normalize = normalize
        if normalize:
            self.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:  # index should only raise ValueError
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str) -> list[int]:
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in regex.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def decode(self, tokens: list[int]) -> str:
        text = "".join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="replace").replace("</w>", " ")
        return text

    def batch_decode(
        self,
        output_ids: torch.Tensor,
        skip_special_tokens: bool = True,
        special_tokens: tuple[str, ...] = ("",),
    ) -> list[str]:
        if skip_special_tokens:
            special_tokens = [self.eos_token, self.bos_token] + list(special_tokens)
        output = []
        for o in output_ids:
            o = o.cpu().numpy()
            text = self.decode(o)
            text = text.split(self.eos_token)[0]
            if skip_special_tokens:
                for tok in special_tokens:
                    text = text.replace(tok, "")
            output.append(text)
        return output

    def __call__(
        self,
        texts: str | list[str],
        append_eos: bool = True,
        cut_max_seq_length: bool = False,
    ) -> torch.Tensor:
        """Returns the tokenized representation of given input string(s)

        Args:
            texts: An input string or a list of input strings to tokenize
            append_eos: Force append end of sentence (aka end of text) token
            cut_max_seq_length: Force cut to max sequence length

        Returns:
            A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.normalize:
            texts = [self.normalizer.normalize_str(text) for text in texts]

        # optionally append bos and eos tokens
        assert not append_eos or self.add_eos_token, "self.add_bos_token must be True if append_eos is True"
        add_eos_token = self.add_eos_token if append_eos else False
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if add_eos_token else []

        all_tokens = [bos_token_id + self.encode(text) + eos_token_id for text in texts]

        if cut_max_seq_length:
            max_seq_length = min(max([len(tokens) for tokens in all_tokens]), self.context_length)
        else:
            max_seq_length = self.context_length

        result = torch.full(
            size=(len(all_tokens), max_seq_length),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.context_length:
                tokens = tokens[: self.context_length]  # Truncate
                if self.add_eos_token:
                    tokens[-1] = eos_token_id[0]
            result[i, : len(tokens)] = torch.tensor(tokens)

        return result


def ignore_tokens_after_special_tokens(
    output_ids: torch.Tensor, special_tokens: list[int], fill_token_id: int
) -> torch.Tensor:
    output_ids_ = torch.clone(output_ids)
    for special_token_id in special_tokens:
        if special_token_id is not None:
            # Create a mask for where to replace elements
            mask = (output_ids_ == special_token_id).cumsum(dim=1).bool()
            # Shift the mask to start replacing after the first 0
            shifted_mask = mask.roll(shifts=1, dims=1)
            shifted_mask[:, 0] = False  # Ensure the first column is not affected by roll
            # Apply the mask to the tensor
            output_ids_[shifted_mask] = fill_token_id
    return output_ids_


class HFTokenizer:
    """HuggingFace tokenizer wrapper"""

    def __init__(
        self,
        tokenizer_name: str,
        context_length: int = 77,
        padding_side: Literal["right", "left"] = "right",
        pad_token: Literal["unk_token", "eos_token", "[PAD]"] | None = None,
        add_eos_token: bool = True,
        add_image_token: bool = False,
        add_eou_token: bool = False,
        lower_case: bool = True,
        max_seq_len: int = -1,
        cut_max_seq_length: bool = False,
        use_fast: bool = True,
        clean_text: bool = True,
        **_: Any,
    ):
        from transformers import AutoTokenizer
        from transformers.tokenization_utils_base import AddedToken

        try:  # only download once
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                padding_side=padding_side,
                add_eos_token=add_eos_token,
                local_files_only=True,
                use_fast=use_fast,
            )
        # if it is not downloaded yet
        except Exception:  # noqa: BLE001
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    padding_side=padding_side,
                    add_eos_token=add_eos_token,
                    use_fast=use_fast,
                )
            except FileNotFoundError:
                from transformers import AutoConfig

                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    config=AutoConfig.from_pretrained(tokenizer_name),
                    padding_side=padding_side,
                    add_eos_token=add_eos_token,
                    use_fast=use_fast,
                )
        # special tokens
        if pad_token is not None:
            if hasattr(tokenizer, pad_token) and getattr(tokenizer, pad_token, None) is not None:
                tokenizer.pad_token = getattr(tokenizer, pad_token)
                if hasattr(tokenizer, f"{pad_token}_id"):
                    tokenizer.pad_token_id = getattr(tokenizer, f"{pad_token}_id")
            else:
                tokenizer.add_special_tokens({"pad_token": pad_token})
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)

        self.image_token_id = None
        self.image_sep_token_id = None
        self.eou_token_id = None

        additional_special_tokens = []
        if add_image_token:
            image_token_ = AddedToken(IMAGE_TOKEN, normalized=False, special=True)
            additional_special_tokens.append(image_token_)
        if add_eou_token:
            eou_token_ = AddedToken(END_OF_UTTERANCE, normalized=False, special=True)
            additional_special_tokens.append(eou_token_)

        tokens_to_add = {"additional_special_tokens": additional_special_tokens}
        tokenizer.add_special_tokens(tokens_to_add)

        if add_image_token:
            self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

        if add_eou_token:
            self.eou_token_id = tokenizer.convert_tokens_to_ids(END_OF_UTTERANCE)

        self.tokenizer = tokenizer
        self.context_length = context_length
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token = self.tokenizer.pad_token
        self.eos_token = self.tokenizer.eos_token
        self.bos_token = self.tokenizer.bos_token
        self.bos_token_id = self.tokenizer.bos_token_id

        self.padding_side = padding_side
        self.lower_case = lower_case
        self.max_seq_len = max_seq_len
        self.cut_max_seq_length = cut_max_seq_length
        self.add_eos_token = add_eos_token
        self.clean_text = clean_text

        self.add_bos_token = tokenizer.add_bos_token if hasattr(tokenizer, "add_bos_token") else False

        if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
            self.pad_token_id = self.tokenizer.pad_token_id
        else:
            self.pad_token_id = self.tokenizer([self.pad_token], return_tensors="pt").input_ids[0][0].item()

        if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            self.eos_token_id = self.tokenizer.eos_token_id
        else:
            self.eos_token_id = self.tokenizer([self.eos_token], return_tensors="pt").input_ids[0][0].item()

    def save_pretrained(self, dest):
        self.tokenizer.save_pretrained(dest)

    def batch_decode(
        self,
        output_ids: torch.Tensor,
        skip_special_tokens: bool = True,
        ignore_after_special_tokens: bool = True,
        **kwargs: Any,
    ) -> list[str]:
        if ignore_after_special_tokens:
            output_ids_ = ignore_tokens_after_special_tokens(
                output_ids,
                special_tokens=[self.eos_token_id, self.eou_token_id],
                fill_token_id=self.eos_token_id,
            )
        else:
            output_ids_ = output_ids
        output_text = self.tokenizer.batch_decode(output_ids_, skip_special_tokens=skip_special_tokens, **kwargs)
        return output_text

    def __call__(
        self,
        texts: str | list[str],
        max_length: int | None = None,
        cut_max_seq_length: bool | None = None,
    ) -> torch.Tensor:
        # same cleaning as for default tokenizer, except lowercasing
        # adding lower (for case-sensitive tokenizers) will make it more robust but less sensitive to nuance
        if isinstance(texts, str):
            texts = [texts]
        if self.clean_text:
            texts = [whitespace_clean(basic_clean(text)) for text in texts]
        if self.lower_case:
            texts = [text.lower() for text in texts]

        max_length = max_length if max_length is not None else self.context_length

        longest = cut_max_seq_length if cut_max_seq_length is not None else self.cut_max_seq_length

        if longest:
            assert self.max_seq_len > 0, "`max_seq_len` has to be positive when `longest` mode is used."
            input_ids = self.tokenizer(
                texts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
            ).input_ids
        else:
            input_ids = self.tokenizer(
                texts,
                return_tensors="pt",
                max_length=max_length,
                padding="max_length",
                truncation=True,
            ).input_ids

        if 0 < self.max_seq_len < input_ids.shape[-1]:
            input_ids = torch.cat((input_ids[:, : self.max_seq_len - 1], input_ids[:, -1:]), dim=-1)

        return input_ids
