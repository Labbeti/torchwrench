from typing import Any, Generic, Literal, NamedTuple, Sequence, overload

import torch
from _typeshed import Incomplete
from pythonwrench import BuiltinNumber
from torch._C import _TensorMeta
from torch.types import Device as Device
from typing_extensions import TypeVar

from torchwrench.core.dtype_enum import DTypeEnum as DTypeEnum
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

T_BuiltinNumber = TypeVar("T_BuiltinNumber", bound=BuiltinNumber, covariant=True)
T_Tensor = TypeVar("T_Tensor", bound="_TensorNDBase", contravariant=True)
T_DType = TypeVar("T_DType", bound="DTypeEnum" | None)
T_NDim = TypeVar("T_NDim", bound=_int)
T_Floating = TypeVar("T_Floating", bound=_bool)
T_Complex = TypeVar("T_Complex", bound=_bool)
T_Signed = TypeVar("T_Signed", bound=_bool)

class _GenericsValues(NamedTuple):
    dtype: torch.dtype | None = ...
    ndim: _int | None = ...
    is_floating_point: _bool | None = ...
    is_complex: _bool | None = ...
    is_signed: _bool | None = ...
    def is_compatible_with_tensor(self, tensor: torch.Tensor) -> _bool: ...
    def is_compatible_with_dtype(self, dtype: torch.dtype) -> _bool: ...
    def is_compatible_with_generic(self, other: _GenericsValues) -> _bool: ...

class _TensorNDMeta(
    _TensorMeta,
    Generic[T_DType, T_NDim, T_BuiltinNumber, T_Floating, T_Complex, T_Signed],
):
    def __instancecheck__(self, instance: Any) -> _bool: ...
    def __subclasscheck__(self, subclass: Any) -> _bool: ...

class _TensorNDBase(
    torch.Tensor,
    Generic[T_DType, T_NDim, T_BuiltinNumber, T_Floating, T_Complex, T_Signed],
):
    @overload
    def __new__(
        cls,
        *dims: _int,
        dtype: DTypeLike = None,
        device: DeviceLike = None,
        memory_format: torch.memory_format | None = None,
        out: torch.Tensor | None = None,
        layout: torch.layout | None = None,
        pin_memory: _bool | None = False,
        requires_grad: _bool | None = False,
    ) -> T_Tensor: ...
    @overload
    def __new__(
        cls,
        data: T_BuiltinNumber | Sequence,
        /,
        *,
        dtype: DTypeLike = None,
        device: DeviceLike = None,
    ) -> T_Tensor: ...
    @overload
    def __init__(
        self,
        *dims: _int,
        dtype: DTypeLike = None,
        device: DeviceLike = None,
        memory_format: torch.memory_format | None = None,
        out: torch.Tensor | None = None,
        layout: torch.layout | None = None,
        pin_memory: _bool | None = False,
        requires_grad: _bool | None = False,
    ) -> None: ...
    @overload
    def __init__(
        self,
        data: T_BuiltinNumber | Sequence,
        /,
        *,
        dtype: DTypeLike = None,
        device: DeviceLike = None,
    ) -> None: ...
    @overload
    def __eq__(self, other: Any) -> BoolTensor: ...
    @overload
    def __getitem__(self, idx: _int) -> Tensor0D: ...
    @overload
    def __getitem__(self, idx: _int) -> Tensor1D: ...
    @overload
    def __getitem__(self, idx: _int) -> Tensor2D: ...
    @overload
    def __getitem__(self, idx: None) -> Tensor1D: ...
    @overload
    def __getitem__(self, idx: None) -> Tensor2D: ...
    @overload
    def __getitem__(self, idx: None) -> Tensor3D: ...
    @overload
    def __getitem__(self, idx: None) -> Tensor: ...
    @overload
    def __getitem__(self, sl: slice) -> T_Tensor: ...
    @overload
    def __getitem__(self, *args) -> Tensor: ...
    @overload
    def __ne__(self, other: Any) -> BoolTensor: ...
    @overload
    def abs(self) -> T_Tensor: ...
    @overload
    def absolute(self) -> T_Tensor: ...
    @overload
    def acos(self) -> T_Tensor: ...
    @overload
    def all(self, dim: Literal[None] = None) -> BoolTensor0D: ...
    @overload
    def all(
        self, dim: _int | tuple[_int, ...], keepdim: _bool = False
    ) -> BoolTensor: ...
    @overload
    def any(self, dim: Literal[None] = None) -> BoolTensor0D: ...
    @overload
    def any(
        self, dim: _int | tuple[_int, ...], keepdim: _bool = False
    ) -> BoolTensor: ...
    @overload
    def bool(self) -> BoolTensor: ...
    @overload
    def contiguous(self) -> T_Tensor: ...
    @overload
    def cpu(self) -> T_Tensor: ...
    @overload
    def cuda(self) -> T_Tensor: ...
    @overload
    def double(self) -> DoubleTensor: ...
    @overload
    def eq(self, other: torch.Tensor | BuiltinNumber) -> BoolTensor: ...
    @overload
    def equal(self, other: torch.Tensor) -> _bool: ...
    @overload
    def float(self) -> FloatTensor: ...
    @overload
    def half(self) -> HalfTensor: ...
    @overload
    def int(self) -> IntTensor: ...
    @overload
    def is_complex(self) -> T_Complex: ...
    @overload
    def is_floating_point(self) -> T_Floating: ...
    @overload
    def is_signed(self) -> T_Signed: ...
    @overload
    def isfinite(self) -> BoolTensor: ...
    @overload
    def isinf(self) -> BoolTensor: ...
    @overload
    def isnan(self) -> BoolTensor: ...
    @overload
    def item(self) -> T_BuiltinNumber: ...
    @overload
    def long(self) -> LongTensor: ...
    @overload
    def mean(self, dim: Literal[None] = None) -> Tensor0D: ...
    @overload
    def mean(self, dim: _int) -> Tensor0D: ...
    @overload
    def mean(self, dim: _int) -> Tensor0D: ...
    @overload
    def mean(self, dim: _int) -> Tensor1D: ...
    @overload
    def mean(self, dim: _int) -> Tensor2D: ...
    @overload
    def mean(self, dim: _int) -> Tensor: ...
    @overload
    def reshape(self, size: tuple[()]) -> Tensor0D: ...
    @overload
    def reshape(self, size: tuple[_int]) -> Tensor1D: ...
    @overload
    def reshape(self, size: tuple[_int, _int]) -> Tensor2D: ...
    @overload
    def reshape(self, size: tuple[_int, _int, _int]) -> Tensor3D: ...
    @overload
    def reshape(self, size: tuple[_int, ...]) -> Tensor: ...
    @overload
    def reshape(self, size0: _int) -> Tensor1D: ...
    @overload
    def reshape(self, size0: _int, size1: _int) -> Tensor2D: ...
    @overload
    def reshape(self, size0: _int, size1: _int, size2: _int) -> Tensor3D: ...
    @overload
    def short(self) -> ShortTensor: ...
    @overload
    def squeeze(self, dim: _int | None = None) -> Tensor: ...
    @overload
    def sum(self, dim: Literal[None] = None) -> Tensor0D: ...
    @overload
    def sum(self, dim: _int) -> Tensor0D: ...
    @overload
    def sum(self, dim: _int) -> Tensor1D: ...
    @overload
    def sum(self, dim: _int) -> Tensor2D: ...
    @overload
    def sum(self, dim: _int | None = None) -> Tensor: ...
    @overload
    def to(
        self,
        dtype: torch.dtype | None = None,
        non_blocking: _bool = False,
        copy: _bool = False,
        *,
        memory_format: torch.memory_format | None = None,
    ) -> T_Tensor: ...
    @overload
    def to(
        self,
        device: Device = None,
        dtype: torch.dtype | None = None,
        non_blocking: _bool = False,
        copy: _bool = False,
        *,
        memory_format: torch.memory_format | None = None,
    ) -> T_Tensor: ...
    @overload
    def to(
        self,
        other: T_Tensor,
        non_blocking: _bool = False,
        copy: _bool = False,
        *,
        memory_format: torch.memory_format | None = None,
    ) -> T_Tensor: ...
    @overload
    def tolist(self) -> Any: ...
    @overload
    def unsqueeze(self, dim: _int) -> Tensor1D: ...
    @overload
    def unsqueeze(self, dim: _int) -> Tensor2D: ...
    @overload
    def unsqueeze(self, dim: _int) -> Tensor3D: ...
    @overload
    def unsqueeze(self, dim: _int) -> Tensor: ...
    @overload
    def view(self, size: tuple[()]) -> Tensor0D: ...
    @overload
    def view(self, size: tuple[_int]) -> Tensor1D: ...
    @overload
    def view(self, size: tuple[_int, _int]) -> Tensor2D: ...
    @overload
    def view(self, size: tuple[_int, _int, _int]) -> Tensor3D: ...
    @overload
    def view(self, size: tuple[_int, ...]) -> Tensor: ...
    @overload
    def view(self, size0: _int) -> Tensor1D: ...
    @overload
    def view(self, size0: _int, size1: _int) -> Tensor2D: ...
    @overload
    def view(self, size0: _int, size1: _int, size2: _int) -> Tensor3D: ...
    @overload
    def view(self, *size: _int) -> Tensor: ...
    @overload
    def view(self, dtype: torch.dtype) -> Tensor: ...
    ndim: T_NDim
    __eq__: Incomplete
    __getitem__: Incomplete
    __ne__: Incomplete
    abs: Incomplete
    absolute: Incomplete
    acos: Incomplete
    all: Incomplete
    any: Incomplete
    bool: Incomplete
    contiguous: Incomplete
    cpu: Incomplete
    cuda: Incomplete
    double: Incomplete
    eq: Incomplete
    equal: Incomplete
    float: Incomplete
    half: Incomplete
    is_complex: Incomplete
    is_floating_point: Incomplete
    is_signed: Incomplete
    isfinite: Incomplete
    isinf: Incomplete
    isnan: Incomplete
    int: Incomplete
    item: Incomplete
    long: Incomplete
    mean: Incomplete
    reshape: Incomplete
    short: Incomplete
    squeeze: Incomplete
    sum: Incomplete
    to: Incomplete
    tolist: Incomplete
    unsqueeze: Incomplete
    view: Incomplete

class Tensor(
    _TensorNDBase[Literal[None], _int, BuiltinNumber, _bool, _bool, _bool]
): ...
class Tensor0D(
    _TensorNDBase[Literal[None], Literal[0], BuiltinNumber, _bool, _bool, _bool]
): ...
class Tensor1D(
    _TensorNDBase[Literal[None], Literal[1], BuiltinNumber, _bool, _bool, _bool]
): ...
class Tensor2D(
    _TensorNDBase[Literal[None], Literal[2], BuiltinNumber, _bool, _bool, _bool]
): ...
class Tensor3D(
    _TensorNDBase[Literal[None], Literal[3], BuiltinNumber, _bool, _bool, _bool]
): ...
class BoolTensor(
    _TensorNDBase[
        Literal[DTypeEnum.bool],
        _int,
        _bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ]
): ...

class BoolTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.bool],
        Literal[0],
        _bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ]
):
    def tolist(self) -> _bool: ...

class BoolTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.bool],
        Literal[1],
        _bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ]
):
    def tolist(self) -> list[_bool]: ...

class BoolTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.bool],
        Literal[2],
        _bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ]
):
    def tolist(self) -> list[list[_bool]]: ...

class BoolTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.bool],
        Literal[3],
        _bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ]
):
    def tolist(self) -> list[list[list[_bool]]]: ...

class ByteTensor(
    _TensorNDBase[
        Literal[DTypeEnum.uint8],
        _int,
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ]
): ...

class ByteTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.uint8],
        Literal[0],
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ]
):
    def tolist(self) -> _int: ...

class ByteTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.uint8],
        Literal[1],
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ]
):
    def tolist(self) -> list[_int]: ...

class ByteTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.uint8],
        Literal[2],
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ]
):
    def tolist(self) -> list[list[_int]]: ...

class ByteTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.uint8],
        Literal[3],
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ]
):
    def tolist(self) -> list[list[list[_int]]]: ...

class CharTensor(
    _TensorNDBase[
        Literal[DTypeEnum.int8],
        _int,
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ]
): ...

class CharTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.int8],
        Literal[0],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> _int: ...

class CharTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.int8],
        Literal[1],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> list[_int]: ...

class CharTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.int8],
        Literal[2],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> list[list[_int]]: ...

class CharTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.int8],
        Literal[3],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> list[list[list[_int]]]: ...

class DoubleTensor(
    _TensorNDBase[
        Literal[DTypeEnum.double],
        _int,
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ]
): ...

class DoubleTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.double],
        Literal[0],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> _float: ...

class DoubleTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.double],
        Literal[1],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> list[_float]: ...

class DoubleTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.double],
        Literal[2],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> list[list[_float]]: ...

class DoubleTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.double],
        Literal[3],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> list[list[list[_float]]]: ...

class FloatTensor(
    _TensorNDBase[
        Literal[DTypeEnum.float],
        _int,
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ]
): ...
class FloatTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.float],
        Literal[0],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ]
): ...

class FloatTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.float],
        Literal[1],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> list[_float]: ...

class FloatTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.float],
        Literal[2],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> list[list[_float]]: ...

class FloatTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.float],
        Literal[3],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> list[list[list[_float]]]: ...

class HalfTensor(
    _TensorNDBase[
        Literal[DTypeEnum.half],
        _int,
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ]
): ...

class HalfTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.half],
        Literal[0],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> _float: ...

class HalfTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.half],
        Literal[1],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> list[_float]: ...

class HalfTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.half],
        Literal[2],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> list[list[_float]]: ...

class HalfTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.half],
        Literal[3],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> list[list[list[_float]]]: ...

class IntTensor(
    _TensorNDBase[
        Literal[DTypeEnum.int],
        _int,
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ]
): ...

class IntTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.int],
        Literal[0],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> _int: ...

class IntTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.int],
        Literal[1],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> list[_int]: ...

class IntTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.int],
        Literal[2],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> list[list[_int]]: ...

class IntTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.int],
        Literal[3],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> list[list[list[_int]]]: ...

class LongTensor(
    _TensorNDBase[
        Literal[DTypeEnum.long],
        _int,
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ]
): ...

class LongTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.long],
        Literal[0],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> _int: ...

class LongTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.long],
        Literal[1],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> list[_int]: ...

class LongTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.long],
        Literal[2],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> list[list[_int]]: ...

class LongTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.long],
        Literal[3],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> list[list[list[_int]]]: ...

class ShortTensor(
    _TensorNDBase[
        Literal[DTypeEnum.short],
        _int,
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ]
): ...

class ShortTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.short],
        Literal[0],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> _int: ...

class ShortTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.short],
        Literal[1],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> list[_int]: ...

class ShortTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.short],
        Literal[2],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> list[list[_int]]: ...

class ShortTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.short],
        Literal[3],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ]
):
    def tolist(self) -> list[list[list[_int]]]: ...

class CFloatTensor(
    _TensorNDBase[
        Literal[DTypeEnum.cfloat],
        _int,
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ]
): ...

class CFloatTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.cfloat],
        Literal[0],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ]
):
    def tolist(self) -> complex: ...

class CFloatTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.cfloat],
        Literal[1],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ]
):
    def tolist(self) -> list[complex]: ...

class CFloatTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.cfloat],
        Literal[2],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ]
):
    def tolist(self) -> list[list[complex]]: ...

class CFloatTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.cfloat],
        Literal[3],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ]
):
    def tolist(self) -> list[list[list[complex]]]: ...

class CHalfTensor(
    _TensorNDBase[
        Literal[DTypeEnum.chalf],
        _int,
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ]
): ...

class CHalfTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.chalf],
        Literal[0],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ]
):
    def tolist(self) -> complex: ...

class CHalfTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.chalf],
        Literal[1],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ]
):
    def tolist(self) -> list[complex]: ...

class CHalfTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.chalf],
        Literal[2],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ]
):
    def tolist(self) -> list[list[complex]]: ...

class CHalfTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.chalf],
        Literal[3],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ]
):
    def tolist(self) -> list[list[list[complex]]]: ...

class CDoubleTensor(
    _TensorNDBase[
        Literal[DTypeEnum.cdouble],
        _int,
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ]
): ...

class CDoubleTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.cdouble],
        Literal[0],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ]
):
    def tolist(self) -> complex: ...

class CDoubleTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.cdouble],
        Literal[1],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ]
):
    def tolist(self) -> list[complex]: ...

class CDoubleTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.cdouble],
        Literal[2],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ]
):
    def tolist(self) -> list[list[complex]]: ...

class CDoubleTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.cdouble],
        Literal[3],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ]
):
    def tolist(self) -> list[list[list[complex]]]: ...

class ComplexFloatingTensor(
    _TensorNDBase[
        Literal[None], _int, complex, Literal[False], Literal[True], Literal[True]
    ]
): ...
class ComplexFloatingTensor0D(
    _TensorNDBase[
        Literal[None], Literal[0], complex, Literal[False], Literal[True], Literal[True]
    ]
): ...
class ComplexFloatingTensor1D(
    _TensorNDBase[
        Literal[None], Literal[1], complex, Literal[False], Literal[True], Literal[True]
    ]
): ...
class ComplexFloatingTensor2D(
    _TensorNDBase[
        Literal[None], Literal[2], complex, Literal[False], Literal[True], Literal[True]
    ]
): ...
class ComplexFloatingTensor3D(
    _TensorNDBase[
        Literal[None], Literal[3], complex, Literal[False], Literal[True], Literal[True]
    ]
): ...
class FloatingTensor(
    _TensorNDBase[
        Literal[None], _int, _float, Literal[True], Literal[False], Literal[True]
    ]
): ...
class FloatingTensor0D(
    _TensorNDBase[
        Literal[None], Literal[0], _float, Literal[True], Literal[False], Literal[True]
    ]
): ...
class FloatingTensor1D(
    _TensorNDBase[
        Literal[None], Literal[1], _float, Literal[True], Literal[False], Literal[True]
    ]
): ...
class FloatingTensor2D(
    _TensorNDBase[
        Literal[None], Literal[2], _float, Literal[True], Literal[False], Literal[True]
    ]
): ...
class FloatingTensor3D(
    _TensorNDBase[
        Literal[None], Literal[3], _float, Literal[True], Literal[False], Literal[True]
    ]
): ...
class SignedIntegerTensor(
    _TensorNDBase[
        Literal[None], _int, _int, Literal[False], Literal[False], Literal[True]
    ]
): ...
class SignedIntegerTensor0D(
    _TensorNDBase[
        Literal[None], Literal[0], _int, Literal[False], Literal[False], Literal[True]
    ]
): ...
class SignedIntegerTensor1D(
    _TensorNDBase[
        Literal[None], Literal[1], _int, Literal[False], Literal[False], Literal[True]
    ]
): ...
class SignedIntegerTensor2D(
    _TensorNDBase[
        Literal[None], Literal[2], _int, Literal[False], Literal[False], Literal[True]
    ]
): ...
class SignedIntegerTensor3D(
    _TensorNDBase[
        Literal[None], Literal[3], _int, Literal[False], Literal[False], Literal[True]
    ]
): ...
class UnsignedIntegerTensor(
    _TensorNDBase[
        Literal[None], _int, _int, Literal[False], Literal[False], Literal[False]
    ]
): ...
class UnsignedIntegerTensor0D(
    _TensorNDBase[
        Literal[None], Literal[0], _int, Literal[False], Literal[False], Literal[False]
    ]
): ...
class UnsignedIntegerTensor1D(
    _TensorNDBase[
        Literal[None], Literal[1], _int, Literal[False], Literal[False], Literal[False]
    ]
): ...
class UnsignedIntegerTensor2D(
    _TensorNDBase[
        Literal[None], Literal[2], _int, Literal[False], Literal[False], Literal[False]
    ]
): ...
class UnsignedIntegerTensor3D(
    _TensorNDBase[
        Literal[None], Literal[3], _int, Literal[False], Literal[False], Literal[False]
    ]
): ...
class IntegralTensor(
    _TensorNDBase[Literal[None], _int, _int, Literal[False], Literal[False], _bool]
): ...
class IntegralTensor0D(
    _TensorNDBase[
        Literal[None], Literal[0], _int, Literal[False], Literal[False], _bool
    ]
): ...
class IntegralTensor1D(
    _TensorNDBase[
        Literal[None], Literal[1], _int, Literal[False], Literal[False], _bool
    ]
): ...
class IntegralTensor2D(
    _TensorNDBase[
        Literal[None], Literal[2], _int, Literal[False], Literal[False], _bool
    ]
): ...
class IntegralTensor3D(
    _TensorNDBase[
        Literal[None], Literal[3], _int, Literal[False], Literal[False], _bool
    ]
): ...
class RealTensor(
    _TensorNDBase[Literal[None], _int, _int, bool, Literal[False], _bool]
): ...
class RealTensor0D(
    _TensorNDBase[Literal[None], Literal[0], _int, bool, Literal[False], _bool]
): ...
class RealTensor1D(
    _TensorNDBase[Literal[None], Literal[1], _int, bool, Literal[False], _bool]
): ...
class RealTensor2D(
    _TensorNDBase[Literal[None], Literal[2], _int, bool, Literal[False], _bool]
): ...
class RealTensor3D(
    _TensorNDBase[Literal[None], Literal[3], _int, bool, Literal[False], _bool]
): ...
