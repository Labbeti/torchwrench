from typing import Hashable, Iterable, Literal, Mapping, Sequence, TypeVar, overload

from pythonwrench.typing import SupportsGetitemLen as SupportsGetitemLen
from torch import Tensor

from torchwrench.core.make import DeviceLike as DeviceLike
from torchwrench.core.make import DTypeLike as DTypeLike
from torchwrench.core.make import as_device as as_device
from torchwrench.core.make import as_dtype as as_dtype
from torchwrench.nn.functional.padding import pad_and_stack_rec as pad_and_stack_rec
from torchwrench.nn.functional.predicate import is_stackable as is_stackable
from torchwrench.nn.functional.transform import as_tensor as as_tensor
from torchwrench.nn.functional.transform import to_item as to_item
from torchwrench.types import BoolTensor1D as BoolTensor1D
from torchwrench.types import BoolTensor2D as BoolTensor2D
from torchwrench.types import LongTensor as LongTensor
from torchwrench.types import LongTensor1D as LongTensor1D
from torchwrench.types import is_number_like as is_number_like
from torchwrench.types import is_tensor_or_array as is_tensor_or_array
from torchwrench.types._typing import TensorOrArray as TensorOrArray

T_Name = TypeVar("T_Name", bound=Hashable)

@overload
def multi_indices_to_multihot(
    indices: Iterable[int],
    num_classes: int,
    *,
    padding_idx: int | None = None,
    device: DeviceLike = None,
) -> BoolTensor1D: ...
@overload
def multi_indices_to_multihot(
    indices: Iterable[Iterable[int]],
    num_classes: int,
    *,
    padding_idx: int | None = None,
    device: DeviceLike = None,
) -> BoolTensor2D: ...
@overload
def multi_indices_to_multihot(
    indices: Iterable,
    num_classes: int,
    *,
    padding_idx: int | None = None,
    device: DeviceLike = None,
    dtype: DTypeLike = ...,
) -> Tensor: ...
@overload
def multi_indices_to_multinames(
    indices: Iterable[int],
    idx_to_name: Mapping[int, T_Name] | SupportsGetitemLen[T_Name],
    *,
    padding_idx: int | None = None,
) -> list[T_Name]: ...
@overload
def multi_indices_to_multinames(
    indices: Iterable[Iterable[int]],
    idx_to_name: Mapping[int, T_Name] | SupportsGetitemLen[T_Name],
    *,
    padding_idx: int | None = None,
) -> list[list[T_Name]]: ...
@overload
def multi_indices_to_multinames(
    indices: Iterable[Iterable[int] | TensorOrArray] | TensorOrArray,
    idx_to_name: Mapping[int, T_Name] | SupportsGetitemLen[T_Name],
    *,
    padding_idx: int | None = None,
) -> list: ...
@overload
def multihot_to_multi_indices(
    multihot: Iterable[bool],
    *,
    keep_tensor: Literal[False] = False,
    padding_idx: int | None = None,
    dim: int = -1,
) -> list[int]: ...
@overload
def multihot_to_multi_indices(
    multihot: Iterable[bool],
    *,
    keep_tensor: Literal[True],
    padding_idx: int | None = None,
    dim: int = -1,
) -> LongTensor1D: ...
@overload
def multihot_to_multi_indices(
    multihot: Iterable[Iterable[bool]],
    *,
    keep_tensor: Literal[False] = False,
    padding_idx: int | None = None,
    dim: int = -1,
) -> list[list[int]]: ...
@overload
def multihot_to_multi_indices(
    multihot: TensorOrArray | Iterable[TensorOrArray],
    *,
    keep_tensor: bool = False,
    padding_idx: int | None = None,
    dim: int = -1,
) -> list | LongTensor: ...
def multihot_to_multinames(
    multihot: TensorOrArray
    | Iterable[TensorOrArray]
    | Iterable[bool]
    | Iterable[Iterable[bool]],
    idx_to_name: Mapping[int, T_Name] | Sequence[T_Name],
    *,
    dim: int = -1,
) -> list: ...
def multinames_to_multi_indices(
    names: list[list[T_Name]], idx_to_name: Mapping[int, T_Name] | Sequence[T_Name]
) -> list[list[int]]: ...
def multinames_to_multihot(
    names: list[list[T_Name]],
    idx_to_name: Mapping[int, T_Name] | Sequence[T_Name],
    *,
    device: DeviceLike = None,
    dtype: DTypeLike = ...,
) -> Tensor: ...
def probs_to_multi_indices(
    probs: TensorOrArray,
    threshold: float | Sequence[float] | TensorOrArray,
    *,
    padding_idx: int | None = None,
    dim: int = -1,
) -> list | LongTensor: ...
def probs_to_multihot(
    probs: TensorOrArray,
    threshold: float | Sequence[float] | TensorOrArray,
    *,
    dim: int = -1,
    device: DeviceLike = None,
    dtype: DTypeLike = ...,
) -> Tensor: ...
def probs_to_multinames(
    probs: TensorOrArray,
    threshold: float | Sequence[float] | TensorOrArray,
    idx_to_name: Mapping[int, T_Name] | Sequence[T_Name],
) -> list[list[T_Name]]: ...
def indices_to_multihot(
    indices: Iterable,
    num_classes: int,
    *,
    padding_idx: int | None = None,
    device: DeviceLike = None,
    dtype: DTypeLike = ...,
) -> Tensor: ...
def indices_to_multinames(
    indices: Iterable[int | Iterable[int] | TensorOrArray] | TensorOrArray,
    idx_to_name: Mapping[int, T_Name] | Sequence[T_Name],
    *,
    padding_idx: int | None = None,
) -> list[list[T_Name]]: ...
def multihot_to_indices(
    multihot: Iterable,
    *,
    keep_tensor: bool = False,
    padding_idx: int | None = None,
    dim: int = -1,
) -> list | LongTensor: ...
def multinames_to_indices(
    names: list[list[T_Name]], idx_to_name: Mapping[int, T_Name] | Sequence[T_Name]
) -> list[list[int]]: ...
def probs_to_indices(
    probs: TensorOrArray,
    threshold: float | Sequence[float] | TensorOrArray,
    *,
    padding_idx: int | None = None,
    dim: int = -1,
) -> list | LongTensor: ...
