from typing import Iterable

from torch import Tensor

from torchwrench.core.make import (
    DeviceLike as DeviceLike,
)
from torchwrench.core.make import (
    DTypeLike as DTypeLike,
)
from torchwrench.core.make import (
    as_device as as_device,
)
from torchwrench.core.make import (
    as_dtype as as_dtype,
)
from torchwrench.types import (
    LongTensor as LongTensor,
)
from torchwrench.types import (
    LongTensor1D as LongTensor1D,
)
from torchwrench.types import (
    T_TensorOrArray as T_TensorOrArray,
)

def masked_mean(
    x: T_TensorOrArray,
    non_pad_mask: T_TensorOrArray,
    *,
    dim: None | int | Iterable[int] = None,
    min_div: float | None = 1.0,
) -> T_TensorOrArray: ...
def masked_sum(
    x: T_TensorOrArray,
    non_pad_mask: T_TensorOrArray,
    *,
    dim: None | int | Iterable[int] = None,
) -> T_TensorOrArray: ...
def masked_equal(x1: Tensor, x2: Tensor, mask: Tensor) -> bool: ...
def generate_square_subsequent_mask(
    size: int, diagonal: int = 0, *, device: DeviceLike = None, dtype: DTypeLike = None
) -> Tensor: ...
def lengths_to_non_pad_mask(
    lengths: Tensor,
    max_len: int | None = None,
    include_end: bool = False,
    *,
    dtype: DTypeLike = None,
) -> Tensor: ...
def lengths_to_pad_mask(
    lengths: Tensor,
    max_len: int | None = None,
    include_end: bool = True,
    *,
    dtype: DTypeLike = None,
) -> Tensor: ...
def non_pad_mask_to_lengths(
    mask: T_TensorOrArray, *, dim: int = -1
) -> T_TensorOrArray: ...
def pad_mask_to_lengths(mask: T_TensorOrArray, *, dim: int = -1) -> T_TensorOrArray: ...
def tensor_to_lengths(
    tensor: Tensor,
    *,
    pad_value: float | None = None,
    end_value: float | None = None,
    dim: int = -1,
) -> LongTensor: ...
def tensor_to_non_pad_mask(
    tensor: Tensor,
    *,
    pad_value: float | None = None,
    end_value: float | None = None,
    include_end: bool = False,
    dtype: DTypeLike = None,
) -> Tensor: ...
def tensor_to_pad_mask(
    tensor: Tensor,
    *,
    pad_value: float | None = None,
    end_value: float | None = None,
    include_end: bool = True,
    dtype: DTypeLike = None,
) -> Tensor: ...
def tensor_to_tensors_list(
    x: Tensor,
    *,
    pad_value: float | None = None,
    end_value: float | None = None,
    non_pad_mask: Tensor | None = None,
    lengths: None | Tensor | list[int] = None,
    dim: int = -1,
) -> list[Tensor]: ...
def tensors_list_to_lengths(tensors: list[Tensor], dim: int = -1) -> LongTensor1D: ...
def ratios_to_lengths(
    ratios: Tensor, max_len: int, dtype: DTypeLike = None
) -> Tensor: ...
def ratios_to_non_pad_mask(
    ratios: Tensor, max_len: int, include_end: bool = False, *, dtype: DTypeLike = None
) -> Tensor: ...
def ratios_to_pad_mask(
    ratios: Tensor, max_len: int, include_end: bool = True, *, dtype: DTypeLike = None
) -> Tensor: ...
def lengths_to_ratios(lengths: Tensor, max_len: int | None = None) -> Tensor: ...
def non_pad_mask_to_ratios(non_pad_mask: Tensor, *, dim: int = -1) -> Tensor: ...
def pad_mask_to_ratios(pad_mask: Tensor, *, dim: int = -1) -> Tensor: ...
