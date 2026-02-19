from torch import Tensor as Tensor

from torchwrench.nn.functional.powerset import (
    build_powerset_mapping as build_powerset_mapping,
)
from torchwrench.nn.functional.powerset import (
    multilabel_to_powerset as multilabel_to_powerset,
)
from torchwrench.nn.functional.powerset import (
    powerset_to_multilabel as powerset_to_multilabel,
)
from torchwrench.types import Tensor2D as Tensor2D
from torchwrench.types import Tensor3D as Tensor3D

from .module import Module as Module

class MultilabelToPowerset(Module):
    def __init__(self, num_classes: int, max_set_size: int) -> None: ...
    def forward(self, multilabel: Tensor) -> Tensor: ...
    @property
    def max_set_size(self) -> int: ...
    @property
    def num_powerset_classes(self) -> int: ...
    @property
    def num_classes(self) -> int: ...

class PowersetToMultilabel(Module):
    def __init__(
        self, num_classes: int, max_set_size: int, soft: bool = False
    ) -> None: ...
    def forward(self, powerset: Tensor, soft: bool | None = None) -> Tensor3D: ...
    @property
    def max_set_size(self) -> int: ...
    @property
    def num_powerset_classes(self) -> int: ...
    @property
    def num_classes(self) -> int: ...
    @property
    def soft(self) -> bool: ...
