from typing import Generic, Mapping

from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch import nn

from torchwrench.core.make import DeviceLike as DeviceLike
from torchwrench.core.make import DTypeLike as DTypeLike
from torchwrench.nn.functional.multilabel import (
    T_Name as T_Name,
)
from torchwrench.nn.functional.multilabel import (
    multi_indices_to_multihot as multi_indices_to_multihot,
)
from torchwrench.nn.functional.multilabel import (
    multi_indices_to_multinames as multi_indices_to_multinames,
)
from torchwrench.nn.functional.multilabel import (
    multihot_to_multi_indices as multihot_to_multi_indices,
)
from torchwrench.nn.functional.multilabel import (
    multihot_to_multinames as multihot_to_multinames,
)
from torchwrench.nn.functional.multilabel import (
    multinames_to_multi_indices as multinames_to_multi_indices,
)
from torchwrench.nn.functional.multilabel import (
    multinames_to_multihot as multinames_to_multihot,
)
from torchwrench.nn.functional.multilabel import (
    probs_to_multi_indices as probs_to_multi_indices,
)
from torchwrench.nn.functional.multilabel import (
    probs_to_multihot as probs_to_multihot,
)
from torchwrench.nn.functional.multilabel import (
    probs_to_multinames as probs_to_multinames,
)
from torchwrench.types import LongTensor as LongTensor

from .module import Module as Module

class MultiIndicesToMultihot(Module):
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
    def forward(self, indices: list[list[int]] | list[Tensor]) -> Tensor: ...
    def extra_repr(self) -> str: ...

class MultiIndicesToMultinames(nn.Module, Generic[T_Name]):
    idx_to_name: Incomplete
    padding_idx: Incomplete
    def __init__(
        self, idx_to_name: Mapping[int, T_Name], *, padding_idx: int | None = None
    ) -> None: ...
    def forward(
        self, indices: list[list[int]] | list[Tensor]
    ) -> list[list[T_Name]]: ...
    def extra_repr(self) -> str: ...

class MultihotToMultiIndices(Module):
    padding_idx: Incomplete
    def __init__(self, *, padding_idx: int | None = None) -> None: ...
    def forward(self, multihot: Tensor) -> list | LongTensor: ...
    def extra_repr(self) -> str: ...

class MultihotToMultinames(nn.Module, Generic[T_Name]):
    idx_to_name: Incomplete
    def __init__(self, idx_to_name: Mapping[int, T_Name]) -> None: ...
    def forward(self, multihot: Tensor) -> list[list[T_Name]]: ...

class MultinamesToMultiIndices(nn.Module, Generic[T_Name]):
    idx_to_name: Incomplete
    def __init__(self, idx_to_name: Mapping[int, T_Name]) -> None: ...
    def forward(self, names: list[list[T_Name]]) -> list[list[int]]: ...

class MultinamesToMultihot(nn.Module, Generic[T_Name]):
    idx_to_name: Incomplete
    device: Incomplete
    dtype: Incomplete
    def __init__(
        self,
        idx_to_name: Mapping[int, T_Name],
        *,
        device: DeviceLike = None,
        dtype: DTypeLike = ...,
    ) -> None: ...
    def forward(self, names: list[list[T_Name]]) -> Tensor: ...
    def extra_repr(self) -> str: ...

class ProbsToMultiIndices(Module):
    threshold: Incomplete
    padding_idx: Incomplete
    def __init__(
        self, threshold: float | Tensor, *, padding_idx: int | None = None
    ) -> None: ...
    def forward(self, probs: Tensor) -> list | LongTensor: ...

class ProbsToMultihot(Module):
    threshold: Incomplete
    device: Incomplete
    dtype: Incomplete
    def __init__(
        self,
        threshold: float | Tensor,
        *,
        device: DeviceLike = None,
        dtype: DTypeLike = ...,
    ) -> None: ...
    def forward(self, probs: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class ProbsToMultinames(nn.Module, Generic[T_Name]):
    threshold: Incomplete
    idx_to_name: Incomplete
    def __init__(
        self, threshold: float | Tensor, idx_to_name: Mapping[int, T_Name]
    ) -> None: ...
    def forward(self, probs: Tensor) -> list[list[T_Name]]: ...

IndicesToMultihot = MultiIndicesToMultihot
IndicesToMultinames = MultiIndicesToMultinames
MultihotToIndices = MultihotToMultiIndices
MultinamesToIndices = MultinamesToMultiIndices
ProbsToIndices = ProbsToMultiIndices
