from typing import Generic, Mapping, Sequence

from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch import nn

from torchwrench.core.make import DeviceLike as DeviceLike
from torchwrench.core.make import DTypeLike as DTypeLike
from torchwrench.nn.functional.multiclass import T_Name as T_Name
from torchwrench.nn.functional.multiclass import index_to_name as index_to_name
from torchwrench.nn.functional.multiclass import index_to_onehot as index_to_onehot
from torchwrench.nn.functional.multiclass import name_to_index as name_to_index
from torchwrench.nn.functional.multiclass import name_to_onehot as name_to_onehot
from torchwrench.nn.functional.multiclass import onehot_to_index as onehot_to_index
from torchwrench.nn.functional.multiclass import onehot_to_name as onehot_to_name
from torchwrench.nn.functional.multiclass import probs_to_index as probs_to_index
from torchwrench.nn.functional.multiclass import probs_to_name as probs_to_name
from torchwrench.nn.functional.multiclass import probs_to_onehot as probs_to_onehot

from .module import Module as Module

class IndexToOnehot(Module):
    num_classes: Incomplete
    padding_idx: Incomplete
    device: Incomplete
    dtype: Incomplete
    def __init__(
        self,
        num_classes: int,
        *,
        padding_idx: int | None = None,
        device: DeviceLike = None,
        dtype: DTypeLike = ...,
    ) -> None: ...
    def forward(self, index: list[int] | Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class IndexToName(nn.Module, Generic[T_Name]):
    idx_to_name: Incomplete
    def __init__(
        self, idx_to_name: Mapping[int, T_Name] | Sequence[T_Name]
    ) -> None: ...
    def forward(self, index: list[int] | Tensor) -> list[T_Name]: ...

class OnehotToIndex(Module):
    dim: Incomplete
    def __init__(self, dim: int = -1) -> None: ...
    def forward(self, onehot: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class OnehotToName(nn.Module, Generic[T_Name]):
    idx_to_name: Incomplete
    dim: Incomplete
    def __init__(
        self, idx_to_name: Mapping[int, T_Name] | Sequence[T_Name], dim: int = -1
    ) -> None: ...
    def forward(self, onehot: Tensor) -> list[T_Name]: ...
    def extra_repr(self) -> str: ...

class NameToIndex(nn.Module, Generic[T_Name]):
    idx_to_name: Incomplete
    def __init__(
        self, idx_to_name: Mapping[int, T_Name] | Sequence[T_Name]
    ) -> None: ...
    def forward(self, name: list[T_Name]) -> Tensor: ...

class NameToOnehot(nn.Module, Generic[T_Name]):
    idx_to_name: Incomplete
    device: Incomplete
    dtype: Incomplete
    def __init__(
        self,
        idx_to_name: Mapping[int, T_Name] | Sequence[T_Name],
        *,
        device: DeviceLike = None,
        dtype: DTypeLike = ...,
    ) -> None: ...
    def forward(self, name: list[T_Name]) -> Tensor: ...
    def extra_repr(self) -> str: ...

class ProbsToIndex(Module):
    dim: Incomplete
    def __init__(self, dim: int = -1) -> None: ...
    def forward(self, probs: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class ProbsToOnehot(Module):
    dim: Incomplete
    device: Incomplete
    dtype: Incomplete
    def __init__(
        self, *, dim: int = -1, device: DeviceLike = None, dtype: DTypeLike = ...
    ) -> None: ...
    def forward(self, probs: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class ProbsToName(nn.Module, Generic[T_Name]):
    idx_to_name: Incomplete
    dim: Incomplete
    def __init__(
        self, idx_to_name: Mapping[int, T_Name] | Sequence[T_Name], dim: int = -1
    ) -> None: ...
    def forward(self, probs: Tensor) -> list[T_Name]: ...
    def extra_repr(self) -> str: ...
