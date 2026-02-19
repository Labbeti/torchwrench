from typing import Iterable

from _typeshed import Incomplete
from torch import Tensor as Tensor

from torchwrench.core.make import GeneratorLike as GeneratorLike
from torchwrench.nn.functional.cropping import (
    CropAlign as CropAlign,
)
from torchwrench.nn.functional.cropping import (
    crop_dim as crop_dim,
)
from torchwrench.nn.functional.cropping import (
    crop_dims as crop_dims,
)

from .module import Module as Module

class CropDim(Module):
    target_length: Incomplete
    align: CropAlign
    dim: Incomplete
    generator: GeneratorLike
    def __init__(
        self,
        target_length: int,
        *,
        align: CropAlign = "left",
        dim: int = -1,
        generator: GeneratorLike = None,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class CropDims(Module):
    target_lengths: Incomplete
    dims: Incomplete
    aligns: CropAlign | Iterable[CropAlign]
    generator: GeneratorLike
    def __init__(
        self,
        target_lengths: Iterable[int],
        *,
        aligns: CropAlign | Iterable[CropAlign] = "left",
        dims: Iterable[int] = (-1,),
        generator: GeneratorLike = None,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...
