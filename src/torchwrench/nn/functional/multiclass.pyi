from typing import (
    Any,
    Callable,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
    TypeVar,
    overload,
)

from torch import Tensor

from torchwrench.core.make import DeviceLike as DeviceLike
from torchwrench.core.make import DTypeLike as DTypeLike
from torchwrench.core.make import as_device as as_device
from torchwrench.core.make import as_dtype as as_dtype
from torchwrench.extras.numpy import np as np
from torchwrench.nn.functional.others import get_ndim as get_ndim
from torchwrench.nn.functional.others import get_shape as get_shape
from torchwrench.nn.functional.transform import to_item as to_item
from torchwrench.types import BoolTensor2D as BoolTensor2D
from torchwrench.types import BoolTensor3D as BoolTensor3D
from torchwrench.types import LongTensor as LongTensor
from torchwrench.types import SupportsIterLen as SupportsIterLen
from torchwrench.types import is_number_like as is_number_like
from torchwrench.types._typing import T_TensorOrArray as T_TensorOrArray
from torchwrench.types._typing import TensorOrArray as TensorOrArray

T_Name = TypeVar("T_Name", bound=Hashable, covariant=True)

@overload
def index_to_onehot(
    index: Iterable[int],
    num_classes: int,
    *,
    padding_idx: int | None = None,
    device: DeviceLike = None,
) -> BoolTensor2D: ...
@overload
def index_to_onehot(
    index: Iterable[Iterable[int]],
    num_classes: int,
    *,
    padding_idx: int | None = None,
    device: DeviceLike = None,
) -> BoolTensor3D: ...
@overload
def index_to_onehot(
    index: Iterable,
    num_classes: int,
    *,
    padding_idx: int | None = None,
    device: DeviceLike = None,
    dtype: DTypeLike = ...,
) -> Tensor: ...
def one_hot(
    tensor: Sequence[int] | TensorOrArray | Sequence,
    num_classes: int,
    *,
    padding_idx: int | None = None,
    device: DeviceLike = None,
    dtype: DTypeLike = ...,
) -> Tensor: ...
def index_to_name(
    index: Sequence[int] | TensorOrArray | Sequence,
    idx_to_name: Mapping[int, T_Name] | Iterable[T_Name],
    *,
    is_number_fn: Callable[[Any], bool] = ...,
) -> list[T_Name]: ...
def onehot_to_index(
    onehot: T_TensorOrArray, *, padding_idx: int | None = None, dim: int = -1
) -> T_TensorOrArray: ...
def onehot_to_name(
    onehot: Tensor,
    idx_to_name: Mapping[int, T_Name] | Iterable[T_Name],
    *,
    dim: int = -1,
) -> list[T_Name]: ...
def name_to_index(
    name: list[T_Name], idx_to_name: Mapping[int, T_Name] | Iterable[T_Name]
) -> Tensor: ...
def name_to_onehot(
    name: list[T_Name],
    idx_to_name: Mapping[int, T_Name] | SupportsIterLen[T_Name],
    *,
    device: DeviceLike = None,
    dtype: DTypeLike = ...,
) -> Tensor: ...
def probs_to_index(probs: Tensor, *, dim: int = -1) -> LongTensor: ...
def probs_to_onehot(
    probs: Tensor, *, dim: int = -1, device: DeviceLike = None, dtype: DTypeLike = ...
) -> Tensor: ...
def probs_to_name(
    probs: Tensor,
    idx_to_name: Mapping[int, T_Name] | Iterable[T_Name],
    *,
    dim: int = -1,
) -> list[T_Name]: ...
