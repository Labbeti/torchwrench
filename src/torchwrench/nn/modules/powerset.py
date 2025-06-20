#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

from torch import Tensor

from torchwrench.nn.functional.powerset import (
    build_powerset_mapping,
    multilabel_to_powerset,
    powerset_to_multilabel,
)
from torchwrench.types import Tensor2D, Tensor3D

from .module import Module


class MultilabelToPowerset(Module):
    """
    Module version of :func:`~torchwrench.nn.functional.powerset.multilabel_to_powerset`.
    """

    def __init__(self, num_classes: int, max_set_size: int) -> None:
        mapping = build_powerset_mapping(num_classes, max_set_size)

        super().__init__()
        self._max_set_size = max_set_size

        self.register_buffer("_mapping", mapping, persistent=False)
        self._mapping: Tensor2D

    def forward(self, multilabel: Tensor) -> Tensor:
        return multilabel_to_powerset(multilabel, mapping=self._mapping)

    @property
    def max_set_size(self) -> int:
        return self._max_set_size

    @property
    def num_powerset_classes(self) -> int:
        return self._mapping.shape[0]

    @property
    def num_classes(self) -> int:
        return self._mapping.shape[1]


class PowersetToMultilabel(Module):
    """
    Module version of :func:`~torchwrench.nn.functional.powerset.powerset_to_multilabel`.
    """

    def __init__(self, num_classes: int, max_set_size: int, soft: bool = False) -> None:
        mapping = build_powerset_mapping(num_classes, max_set_size)

        super().__init__()
        self._max_set_size = max_set_size
        self._soft = soft

        self.register_buffer("_mapping", mapping, persistent=False)
        self._mapping: Tensor2D

    def forward(self, powerset: Tensor, soft: Optional[bool] = None) -> Tensor3D:
        if soft is None:
            soft = self._soft
        return powerset_to_multilabel(powerset, soft=soft, mapping=self._mapping)

    @property
    def max_set_size(self) -> int:
        return self._max_set_size

    @property
    def num_powerset_classes(self) -> int:
        return self._mapping.shape[0]

    @property
    def num_classes(self) -> int:
        return self._mapping.shape[1]

    @property
    def soft(self) -> bool:
        return self._soft
