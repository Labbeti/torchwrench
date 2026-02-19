from typing import Generic, NamedTuple

from torch import Tensor
from torch.return_types import max as max
from torch.return_types import min as min
from torch.return_types import sort as sort
from torch.return_types import topk as topk
from typing_extensions import TypeVar

__all__ = ["max", "min", "sort", "topk", "shape", "ndim", "top_p"]

T = TypeVar("T")
T_Values = TypeVar("T_Values", bound=Tensor, covariant=True)
T_Indices = TypeVar("T_Indices", bound=Tensor, covariant=True)

class _namedtuple_values_indices(tuple, Generic[T_Values, T_Indices]):
    @property
    def values(self) -> T_Values: ...
    @property
    def indices(self) -> T_Indices: ...

class max(_namedtuple_values_indices[Tensor, Tensor]): ...
class min(_namedtuple_values_indices[Tensor, Tensor]): ...
class sort(_namedtuple_values_indices[Tensor, Tensor]): ...
class topk(_namedtuple_values_indices[Tensor, Tensor]): ...

class _indicator_base(NamedTuple):
    valid: bool
    shape: T

class shape(_indicator_base, Generic[T]): ...
class ndim(_indicator_base, Generic[T]): ...

class shape(NamedTuple, Generic[T]):
    valid: bool
    shape: T

class ndim(NamedTuple, Generic[T]):
    valid: bool
    ndim: T

class top_p(_namedtuple_values_indices): ...
