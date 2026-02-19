from typing import Sequence, overload

import torch
from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch.types import Number as Number

from torchwrench.nn.functional.make import DTypeLike as DTypeLike
from torchwrench.nn.functional.make import as_dtype as as_dtype
from torchwrench.utils import return_types as return_types

from .module import Module as Module

class Abs(Module):
    def forward(self, x: Tensor) -> Tensor: ...

class Angle(Module):
    def forward(self, x: Tensor) -> Tensor: ...

class Exp(Module):
    def forward(self, x: Tensor) -> Tensor: ...

class Exp2(Module):
    def forward(self, x: Tensor) -> Tensor: ...

class FFT(Module):
    def forward(self, x: Tensor) -> Tensor: ...

class IFFT(Module):
    def forward(self, x: Tensor) -> Tensor: ...

class Imag(Module):
    return_zeros: Incomplete
    def __init__(self, *, return_zeros: bool = False) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class Interpolate(Module):
    size: Incomplete
    scale_factor: Incomplete
    mode: Incomplete
    align_corners: Incomplete
    recompute_scale_factor: Incomplete
    antialias: Incomplete
    def __init__(
        self,
        size: int | tuple[int, ...] | None = None,
        scale_factor: float | tuple[float, ...] | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
        recompute_scale_factor: bool | None = None,
        antialias: bool = False,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class Log(Module):
    def forward(self, x: Tensor) -> Tensor: ...

class Log10(Module):
    def forward(self, x: Tensor) -> Tensor: ...

class Log2(Module):
    def forward(self, x: Tensor) -> Tensor: ...

class Max(Module):
    dim: Incomplete
    return_values: Incomplete
    return_indices: Incomplete
    keepdim: Incomplete
    def __init__(
        self,
        dim: int | None = None,
        keepdim: bool = False,
        *,
        return_values: bool = True,
        return_indices: bool | None = None,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor | return_types.max: ...
    def extra_repr(self) -> str: ...

class Mean(Module):
    dim: Incomplete
    keepdim: Incomplete
    dtype: Incomplete
    def __init__(
        self, dim: int | None = None, keepdim: bool = False, dtype: DTypeLike = None
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class Min(Module):
    dim: Incomplete
    return_values: Incomplete
    return_indices: Incomplete
    keepdim: Incomplete
    def __init__(
        self,
        dim: int | None = None,
        keepdim: bool = False,
        *,
        return_values: bool = True,
        return_indices: bool | None = None,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor | return_types.min: ...
    def extra_repr(self) -> str: ...

class Normalize(Module):
    p: Incomplete
    dim: Incomplete
    eps: Incomplete
    def __init__(self, p: float = 2.0, dim: int = 1, eps: float = 1e-12) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class Permute(Module):
    dims: Incomplete
    def __init__(self, *args: int) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class Pow(Module):
    exponent: Incomplete
    def __init__(self, exponent: Number | Tensor) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class Real(Module):
    def forward(self, x: Tensor) -> Tensor: ...

class Repeat(Module):
    repeats: Incomplete
    def __init__(self, *repeats: int) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class RepeatInterleave(Module):
    repeats: Incomplete
    dim: Incomplete
    output_size: Incomplete
    def __init__(
        self, repeats: int | Tensor, dim: int, output_size: int | None = None
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class Reshape(Module):
    shape: Incomplete
    def __init__(self, *shape: int) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class Sort(Module):
    dim: Incomplete
    descending: Incomplete
    return_values: Incomplete
    return_indices: Incomplete
    def __init__(
        self,
        dim: int = -1,
        descending: bool = False,
        *,
        return_values: bool = True,
        return_indices: bool = True,
    ) -> None: ...
    def forward(self, x: Tensor) -> return_types.sort | Tensor: ...

class TensorTo(Module):
    kwargs: Incomplete
    def __init__(self, **kwargs) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class ToList(Module):
    def forward(self, x: Tensor) -> list: ...

class Transpose(Module):
    dim0: Incomplete
    dim1: Incomplete
    copy: Incomplete
    def __init__(self, dim0: int, dim1: int, copy: bool = False) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class View(Module):
    @overload
    def __init__(self, dtype: torch.dtype, /) -> None: ...
    @overload
    def __init__(self, size: Sequence[int], /) -> None: ...
    @overload
    def __init__(self, *size: int) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
