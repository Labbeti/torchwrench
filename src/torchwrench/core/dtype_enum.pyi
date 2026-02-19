from typing import Any, Final

import torch
from _typeshed import Incomplete
from pythonwrench.enum import StrEnum

TORCH_DTYPES: Final[dict[str, torch.dtype]]

class DTypeEnum(StrEnum):
    float16: Incomplete
    float32: Incomplete
    float64: Incomplete
    int16: Incomplete
    int32: Incomplete
    int64: Incomplete
    complex32: Incomplete
    complex64: Incomplete
    complex128: Incomplete
    half = float16
    float = float32
    double = float64
    short = int16
    int = int32
    long = int64
    chalf = complex32
    cfloat = complex64
    cdouble = complex128
    bfloat16: Incomplete
    bool: Incomplete
    int8: Incomplete
    qint8: Incomplete
    qint32: Incomplete
    quint4x2: Incomplete
    quint8: Incomplete
    uint8: Incomplete
    quint2x4: Incomplete
    uint16: Incomplete
    uint32: Incomplete
    uint64: Incomplete
    @classmethod
    def default(cls) -> DTypeEnum: ...
    @classmethod
    def from_dtype(cls, dtype: torch.dtype) -> DTypeEnum: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def is_floating_point(self) -> _bool: ...
    @property
    def is_complex(self) -> _bool: ...
    @property
    def is_signed(self) -> _bool: ...
    @property
    def itemsize(self) -> _int: ...
    def to_real(self) -> DTypeEnum: ...
    def to_complex(self) -> DTypeEnum: ...
    def __eq__(self, other: Any) -> _bool: ...
    def __hash__(self) -> _int: ...

def torch_dtype_to_str(dtype: torch.dtype) -> str: ...
def str_to_torch_dtype(dtype: str) -> torch.dtype: ...
def torch_dtype_to_enum_dtype(dtype: torch.dtype) -> DTypeEnum: ...
def str_to_enum_dtype(dtype: str) -> DTypeEnum: ...
def enum_dtype_to_str(dtype: DTypeEnum) -> str: ...
def enum_dtype_to_torch_dtype(dtype: DTypeEnum) -> torch.dtype: ...
