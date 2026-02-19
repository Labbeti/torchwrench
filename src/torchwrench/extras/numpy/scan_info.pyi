from dataclasses import dataclass
from typing import Any, Generic, Iterable, TypeVar

import pythonwrench as pw
import torch
from pythonwrench import BuiltinScalar as BuiltinScalar

from torchwrench.extras.numpy.definitions import (
    ACCEPTED_NUMPY_DTYPES as ACCEPTED_NUMPY_DTYPES,
)
from torchwrench.extras.numpy.definitions import np as np

T_Invalid = TypeVar("T_Invalid", covariant=True)
T_EmptyNp = TypeVar("T_EmptyNp", covariant=True)
T_EmptyTorch = TypeVar("T_EmptyTorch", covariant=True)

class InvalidTorchDType(metaclass=pw.Singleton): ...

@dataclass(frozen=True)
class ShapeDTypeInfo(Generic[T_Invalid, T_EmptyTorch, T_EmptyNp]):
    shape: tuple[int, ...]
    torch_dtype: torch.dtype | T_Invalid | T_EmptyTorch
    numpy_dtype: np.dtype | T_EmptyNp
    valid_shape: bool
    @property
    def fill_value(self) -> BuiltinScalar: ...
    @property
    def get_ndim(self) -> int: ...
    @property
    def kind(self) -> str: ...

def scan_shape_dtypes(
    x: Any,
    *,
    accept_heterogeneous_shape: bool = False,
    empty_torch: T_EmptyTorch = None,
    empty_np: T_EmptyNp = ...,
) -> ShapeDTypeInfo[InvalidTorchDType, T_EmptyTorch, T_EmptyNp]: ...
def scan_torch_dtype(
    x: Any, *, invalid: T_Invalid = ..., empty: T_EmptyTorch = None
) -> torch.dtype | T_Invalid | T_EmptyTorch: ...
def scan_numpy_dtype(x: Any, *, empty: T_EmptyNp = ...) -> np.dtype | T_EmptyNp: ...
def torch_dtype_to_numpy_dtype(dtype: torch.dtype) -> np.dtype: ...
def numpy_dtype_to_torch_dtype(
    dtype: np.dtype, *, invalid: T_Invalid = ...
) -> torch.dtype | T_Invalid: ...
def numpy_dtype_to_fill_value(dtype: Any) -> BuiltinScalar: ...
def merge_numpy_dtypes(
    dtypes: Iterable[np.dtype | T_EmptyNp], *, empty: T_EmptyNp = ...
) -> np.dtype | T_EmptyNp: ...
def merge_torch_dtypes(
    dtypes: Iterable[torch.dtype | T_Invalid | T_EmptyNp],
    *,
    invalid: T_Invalid = ...,
    empty: T_EmptyNp = None,
) -> torch.dtype | T_Invalid | T_EmptyNp: ...
def get_default_numpy_dtype() -> np.dtype: ...
