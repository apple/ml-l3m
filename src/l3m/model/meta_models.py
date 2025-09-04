# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import abc
from typing import Any

import torch
import torch.nn as nn

from l3m.constants.typing import DATA_DICT

__all__ = [
    "ReadWriteClass",
    "ReadWriteBlock",
    "GenericBlock",
    "MetaModel",
    "InSeriesMetaModels",
    "NoGradModel",
]


class ReadWriteClass(abc.ABC):
    """Provides an ABC for making classes follow the read/write to/from data_dict pattern.

    Args:
        read_key: Where to read from the data_dict.
        write_key: Where to write to the data_dict.
    """

    def __init__(
        self,
        *,
        read_key: str | list[str] = None,
        write_key: str | list[str] = None,
    ):
        self.read_key = read_key
        self.write_key = write_key

    @abc.abstractmethod
    def __call__(self, data_dict: DATA_DICT) -> DATA_DICT: ...


class ReadWriteBlock(nn.Module, abc.ABC):
    """Provides a building block for nn.Modules that support the read/write to/from data_dict pattern.

    Args:
        read_key: Where to read from the data_dict.
        write_key: Where to write to the data_dict.
    """

    def __init__(
        self,
        *,
        read_key: str | list[str] = None,
        write_key: str | list[str] = None,
    ):
        super().__init__()
        self.read_key = read_key
        self.write_key = write_key
        self.generation_mode = False

    def generation(self, mode: bool = True) -> "ReadWriteBlock":
        self.generation_mode = mode
        for module in self.children():
            if isinstance(module, ReadWriteBlock):
                module.generation(mode)
            else:
                for submodule in module.children():
                    if isinstance(submodule, ReadWriteBlock):
                        submodule.generation(mode)
        return self

    @abc.abstractmethod
    def forward(self, data_dict: DATA_DICT) -> DATA_DICT: ...


class GenericBlock(ReadWriteBlock):
    """Wraps a simple nn.Module with the data_dict interface.

    Args:
        module: Pytorch module to wrap.
    """

    def __init__(self, module: nn.Module, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.module = module

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        data_dict[self.write_key] = self.module(data_dict[self.read_key])
        return data_dict


class MetaModel(ReadWriteBlock):
    """Creates a meta model that supports four groups of submodels.
    The execution order is preprocessor -> trunk -> postprocessor -> head.

    This is highly flexible and allows setting parts as nn.Identity.
    It also allows setting submodels as MetaModels themselves.

    All the nn.Modules will receive as input a data_dict of format dict[str, Any].
    They should use this dictionary to read/write from in order to communicate with other
    parts of the pipeline, e.g. from dataloading to the model to the loss.

    Args:
        preprocessor: the first submodel, e.g. tokenzier or patch embedding layer
        trunk: second submodel, e.g. the actual transformer.
        postprocessor: third submodel, e.g. a token selection layer.
        head: forth submodel, e.g. a classifier head.
    """

    def __init__(
        self,
        preprocessor: nn.Module = None,
        trunk: nn.Module = None,
        postprocessor: nn.Module = None,
        head: nn.Module = None,
    ):
        super().__init__()
        self.preprocessor = preprocessor if preprocessor else nn.Identity()
        self.trunk = trunk if trunk else nn.Identity()
        self.postprocessor = postprocessor if postprocessor else nn.Identity()
        self.head = head if head else nn.Identity()

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        data_dict = self.preprocessor(data_dict)
        data_dict = self.trunk(data_dict)
        data_dict = self.postprocessor(data_dict)
        data_dict = self.head(data_dict)
        return data_dict


class InSeriesMetaModels(ReadWriteBlock):
    """Allows executing MetaModels in sequence or directly executing an specific MetaModel.

    Args:
        models: Dict of MetaModels, should be ordered in the desired execution order.
        order: Overrides the model execution order to follow this if provided.
    """

    def __init__(
        self,
        models: dict[str, MetaModel],
        order: list[str] = None,
    ):
        super().__init__()

        if order:
            assert set(order) == set(models.keys()), "Order needs to be a 1 to 1 mapping for the models."

            self.models = nn.ModuleDict({name: models[name] for name in order})
        else:
            self.models = nn.ModuleDict(models)

    def _forward(self, data_dict: DATA_DICT, model_name: str | None = None) -> DATA_DICT:
        data_dict = self.models[model_name](data_dict)
        return data_dict

    def forward(
        self,
        data_dict: DATA_DICT,
        model_name: str | None = None,
    ) -> DATA_DICT:
        if model_name:
            data_dict = self._forward(data_dict, model_name=model_name)
        else:
            for model in self.models.values():
                data_dict = model(data_dict)
        return data_dict


class NoGradModel(ReadWriteBlock):
    def __init__(self, model: MetaModel):
        super().__init__()
        self.model = model
        for p in model.parameters():
            p.requires_grad = False

    def forward(self, data_dict: DATA_DICT) -> DATA_DICT:
        with torch.no_grad():
            data_dict = self.model(data_dict)

        return data_dict
