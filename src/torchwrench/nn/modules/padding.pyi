from typing import Iterable, Literal

from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch.types import Number as Number

from torchwrench.core.make import (
    DeviceLike as DeviceLike,
)
from torchwrench.core.make import (
    DTypeLike as DTypeLike,
)
from torchwrench.core.make import (
    GeneratorLike as GeneratorLike,
)
from torchwrench.nn.functional.padding import (
    PadAlign as PadAlign,
)
from torchwrench.nn.functional.padding import (
    PadMode as PadMode,
)
from torchwrench.nn.functional.padding import (
    PadValue as PadValue,
)
from torchwrench.nn.functional.padding import (
    pad_and_stack_rec as pad_and_stack_rec,
)
from torchwrench.nn.functional.padding import (
    pad_dim as pad_dim,
)
from torchwrench.nn.functional.padding import (
    pad_dims as pad_dims,
)

from .module import Module as Module

class PadDim(Module):
    target_length: Incomplete
    dim: Incomplete
    align: PadAlign
    pad_value: Incomplete
    mode: PadMode
    generator: GeneratorLike
    def __init__(
        self,
        target_length: int,
        *,
        dim: int = -1,
        align: PadAlign = "left",
        pad_value: PadValue = 0.0,
        mode: PadMode = "constant",
        generator: GeneratorLike = None,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class PadDims(Module):
    target_lengths: Incomplete
    dims: Iterable[int] | None | Literal["auto"]
    aligns: PadAlign | Iterable[PadAlign]
    pad_value: Incomplete
    mode: PadMode
    generator: GeneratorLike
    def __init__(
        self,
        target_lengths: Iterable[int],
        *,
        dims: Iterable[int] | None | Literal["auto"] = None,
        aligns: PadAlign | Iterable[PadAlign] = "left",
        pad_value: PadValue = 0.0,
        mode: PadMode = "constant",
        generator: GeneratorLike = None,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class PadAndStackRec(Module):
    pad_value: Incomplete
    align: PadAlign
    device: Incomplete
    dtype: Incomplete
    def __init__(
        self,
        pad_value: Number = 0,
        *,
        align: PadAlign = "left",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...
    def forward(self, sequence: Tensor | int | float | tuple | list) -> Tensor: ...
    def extra_repr(self) -> str: ...
