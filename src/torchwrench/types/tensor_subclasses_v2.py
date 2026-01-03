#!/usr/bin/env python
# -*- coding: utf-8 -*-

from builtins import bool as _bool
from dataclasses import dataclass
from enum import auto
from typing import (
    Any,
    Generic,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
    overload,
)

import torch
from pythonwrench.enum import StrEnum
from pythonwrench.typing.classes import BuiltinNumber
from torch._C import _TensorMeta
from typing_extensions import Self, Type, TypeAlias, TypeVar

from torchwrench.core.dtype_enum import DTypeEnum, _bool, _int
from torchwrench.core.make import DeviceLike, DTypeLike, as_device, as_dtype

_0DShape = Tuple[()]
_1DShape = Tuple[int]
_2DShape = Tuple[int, int]
_3DShape = Tuple[int, int, int]
_4DShape = Tuple[int, int, int, int]


class DeviceEnum(StrEnum):
    cuda = auto()
    cpu = auto()


_DefaultShape: TypeAlias = None
_DefaultDType: TypeAlias = None
_DefaultDevice: TypeAlias = None

_DefaultNDim: TypeAlias = None

_ShapeGenericType: TypeAlias = Union[Tuple[int, ...], _DefaultShape]
_DTypeGenericType: TypeAlias = Union[DTypeEnum, _DefaultDType]
_DeviceGenericType: TypeAlias = Union[DeviceEnum, _DefaultDevice]


T_Shape = TypeVar(
    "T_Shape",
    bound=_ShapeGenericType,
    default=_DefaultShape,
    covariant=True,
)
T_DType = TypeVar(
    "T_DType",
    bound=_DTypeGenericType,
    default=_DefaultDType,
    covariant=True,
)
T_Device = TypeVar(
    "T_Device",
    bound=_DeviceGenericType,
    default=_DefaultDevice,
    covariant=True,
)

T_Shape2 = TypeVar(
    "T_Shape2",
    bound=_ShapeGenericType,
    default=_DefaultShape,
    covariant=True,
)
T_DType2 = TypeVar(
    "T_DType2",
    bound=_DTypeGenericType,
    default=_DefaultDType,
    covariant=True,
)
T_Device2 = TypeVar(
    "T_Device2",
    bound=_DeviceGenericType,
    default=_DefaultDevice,
    covariant=True,
)


def get_typed_shape_to_shape(
    typed_shape: _ShapeGenericType,
) -> Optional[Tuple[Union[int, Type[int]], ...]]:
    if typed_shape is None:
        return None
    else:
        return tuple(
            get_args(s)[0] if (get_origin(s) is Literal) else s
            for s in get_args(typed_shape)
        )


@dataclass
class _TTensorTypes:
    shape: Union[Type[Tuple[int, ...]], _DefaultShape]
    dtype: Union[Type[DTypeEnum], _DefaultDType]
    device: Union[Type[DeviceEnum], _DefaultDevice]

    def is_compatible_with_tensor(self, x: torch.Tensor) -> _bool:
        return (
            self.is_compatible_with_shape(x.shape)
            and self.is_compatible_with_dtype(x.dtype)
            and self.is_compatible_with_device(x.device)
        )

    def is_compatible_with_shape(self, shape: Tuple[int, ...]) -> _bool:
        if self.shape is None:
            return True

        if len(get_args(self.shape)) != len(shape):
            return False

        valid = [
            (isinstance(s1, type) and issubclass(s1, int)) or s1 == s2
            for s1, s2 in zip(get_args(self.shape), shape)
        ]
        return all(valid)

    def is_compatible_with_dtype(self, dtype: Union[torch.dtype, DTypeEnum]) -> _bool:
        """Check if the dtype is compatible with the generic dtype."""
        if self.dtype is None or dtype is None:
            return True

        if isinstance(dtype, torch.dtype):
            dtype_enum = DTypeEnum.from_dtype(dtype)
        else:
            dtype_enum = dtype

        return self.dtype == dtype_enum

    def is_compatible_with_device(self, device: Union[torch.device, str, int]) -> _bool:
        if self.device is None:
            return True

        return as_device(self.device) == as_device(device)

    def is_compatible_with_tensor_type(self, x: Self) -> _bool:
        return (
            (x.shape is None or self.is_compatible_with_shape(x.shape))
            and (x.dtype is None or self.is_compatible_with_dtype(x.dtype))
            and (x.device is None or self.is_compatible_with_device(x.device))
        )

    @property
    def ndim(self) -> Union[_int, _DefaultNDim]:
        """Get the number of dimensions from the shape generic parameter."""
        if self.shape is None:
            return None

        generic_shape_args = get_args(self.shape)
        if len(generic_shape_args) == 2 and generic_shape_args[1] is ...:
            return None

        if isinstance(self.shape, tuple):
            return len(self.shape)

        msg = f"Internal error: cannot get ndim for {self.shape=}."
        raise NotImplementedError(msg)


def _get_generics(cls_or_instance: Any) -> _TTensorTypes:
    """Get the generic parameters of a TTensor subclass or instance."""
    if not isinstance(cls_or_instance, type):
        cls_or_instance = type(cls_or_instance)
    return _get_generics_from_type(cls_or_instance)


def _get_generics_from_type(cls: type) -> _TTensorTypes:
    """Get the generic parameters of a TTensor subclass."""

    if get_origin(cls) is TTensor:
        args = get_args(cls)
    elif hasattr(cls, "__orig_bases__"):
        orig_bases = cls.__orig_bases__  # type: ignore
        args = None
        for base in orig_bases:
            if get_origin(base) is TTensor:
                args = get_args(base)
                break
        if args is None:
            msg = f"Invalid argument {cls=}. (expected TTensor or subclass or TTensor)"
            raise TypeError(msg)
    else:
        msg = f"Invalid argument {cls=}. (expected TTensor or subclass or TTensor)"
        raise TypeError(msg)

    # breakpoint()
    # origin = get_origin(cls_or_instance)
    # if origin is None:
    #     origin = cls_or_instance

    # if origin is not _TTensorMeta:
    #     msg = f"Expected a subclass or instance of _TTensorMeta, but got {cls_or_instance}."
    #     raise TypeError(msg)

    if len(args) != 3:
        msg = (
            f"Expected 3 generic parameters for TTensor, but got {len(args)} in {cls}."
        )
        raise TypeError(msg)

    shape_arg, dtype_arg, device_arg = args
    return _TTensorTypes(
        shape=shape_arg,
        dtype=dtype_arg,
        device=device_arg,
    )


class _TTensorMeta(_TensorMeta):
    # def __instancecheck__(self, instance: Any) -> _bool:
    #     """Called method to check isinstance(instance, self)"""
    #     if not isinstance(instance, torch.Tensor):
    #         return False

    #     orig_bases: tuple = self.__orig_bases__  # type: ignore
    #     # breakpoint()
    #     raise NotImplementedError

    # def __subclasscheck__(self, subclass: Any) -> _bool:
    #     """Called method to check issubclass(subclass, cls)"""
    #     orig_bases: tuple = self.__orig_bases__  # type: ignore
    #     gen = _get_generics_from_type(self)
    #     breakpoint()
    #     raise NotImplementedError

    def __instancecheck__(self, instance: Any) -> _bool:
        """Called method to check isinstance(instance, TTensor)"""
        if not isinstance(instance, torch.Tensor):
            return False

        gen = _get_generics_from_type(self)
        return gen.is_compatible_with_tensor(instance)

    def __subclasscheck__(self, subclass: Any) -> _bool:
        """Called method to check issubclass(subclass, TTensor)"""
        self_generics = _get_generics_from_type(self)
        other_generics = _get_generics_from_type(subclass)
        return self_generics.is_compatible_with_tensor_type(other_generics)


class _ndim_descriptor:
    @overload
    def __get__(
        self,
        instance: "TTensor[_0DShape, Any, Any]",
        owner: Any,
    ) -> Literal[0]: ...
    @overload
    def __get__(
        self,
        instance: "TTensor[_1DShape, Any, Any]",
        owner: Any,
    ) -> Literal[1]: ...
    @overload
    def __get__(
        self,
        instance: "TTensor[_2DShape, Any, Any]",
        owner: Any,
    ) -> Literal[2]: ...
    @overload
    def __get__(
        self,
        instance: "TTensor[_3DShape, Any, Any]",
        owner: Any,
    ) -> Literal[3]: ...
    @overload
    def __get__(
        self,
        instance: "TTensor[_4DShape, Any, Any]",
        owner: Any,
    ) -> Literal[4]: ...
    @overload
    def __get__(
        self,
        instance: "TTensor[Any, Any, Any]",
        owner: Any,
    ) -> int: ...

    def __get__(self, instance: object, owner: Any) -> int:
        raise NotImplementedError


class TTensor(
    Generic[T_Shape, T_DType, T_Device],
    torch.Tensor,
    metaclass=_TTensorMeta,
):
    _DEFAULT_DTYPE: Optional[DTypeEnum] = None

    def __new__(
        cls: "Type[TTensor[T_Shape, T_DType, T_Device]]",
        *args: Any,
        dtype: DTypeLike = None,
        device: DeviceLike = None,
        memory_format: Union[torch.memory_format, None] = None,
        out: Union[torch.Tensor, None] = None,
        layout: Union[torch.layout, None] = None,
        pin_memory: Union[_bool, None] = False,
        requires_grad: Union[_bool, None] = False,
    ) -> "TTensor[T_Shape, T_DType, T_Device]":
        dtype = as_dtype(dtype)
        device = as_device(device)

        gen = _get_generics(cls)  # type: ignore
        cls_dtype = gen.dtype
        cls_ndim = gen.ndim
        cls_shape = gen.shape

        # Sanity checks for dtype
        if dtype is None:
            if cls_dtype is not None:
                dtype = cls_dtype
            elif cls._DEFAULT_DTYPE is not None:
                dtype = cls._DEFAULT_DTYPE.dtype

        elif cls_dtype is None:
            if not gen.is_compatible_with_dtype(dtype):
                msg = f"Invalid argument {dtype=} for {cls.__name__}."
                raise ValueError(msg)

        elif dtype == cls_dtype:
            pass
        else:
            msg = (
                f"Invalid argument {dtype=} for {cls.__name__}. (expected {cls_dtype})"
            )
            raise ValueError(msg)

        # Sanity checks for data and ndim
        is_int_args = all(isinstance(arg, _int) for arg in args)
        if cls_ndim is None:
            if len(args) == 0:
                size = (0,)
                data = None
            elif is_int_args:
                size = args
                data = None
            elif len(args) == 1 and not isinstance(args[0], _int):
                size = None
                data = args[0]
            else:
                msg = f"Invalid arguments {args=}. (expected only ints or one sequence of data)"
                raise ValueError(msg)

        elif is_int_args and cls_ndim == len(args):
            if not gen.is_compatible_with_shape(args):
                msg = f"Invalid argument {args=} for {cls.__name__}. (expected shape {cls_shape})"
                raise ValueError(msg)

            size = args
            data = None

        elif len(args) == 1:
            size = None
            data = args[0]
        elif len(args) == 0:
            size = [0] * cls_ndim
            data = None
        else:
            msg = f"Invalid arguments {args=}. (expected {cls_ndim} ints or one sequence of data)"
            raise ValueError(msg)
        del args

        # Supports older PyTorch versions
        if layout is None:
            layout = torch.strided
        if pin_memory is None:
            pin_memory = False
        if requires_grad is None:
            requires_grad = False

        if data is not None:
            result = torch.as_tensor(
                data=data,
                dtype=dtype,  # type: ignore
                device=device,
            )
            if cls_ndim is not None and result.ndim != cls_ndim:
                msg = f"Invalid number of dimension(s) for argument data in {cls.__name__}. (found {result.ndim} but expected {cls_ndim})"
                raise ValueError(msg)
            return result  # type: ignore

        elif size is not None:
            return torch.empty(
                size,
                dtype=dtype,  # type: ignore
                device=device,
                memory_format=memory_format,
                out=out,
                layout=layout,
                pin_memory=pin_memory,
                requires_grad=requires_grad,
            )  # type: ignore

        else:
            msg = f"Internal error: found {data=} and {size=} in {cls.__name__}."
            raise RuntimeError(msg)

    @overload
    def __eq__(self, other: Any) -> "BoolTensor":  # type: ignore
        ...

    @overload
    def __getitem__(  # type: ignore
        self: "TTensor[_2DShape, T_DType, T_Device]",
        idx: _int,
        /,
    ) -> "TTensor[_1DShape, T_DType, T_Device]": ...

    @overload
    def __getitem__(  # type: ignore
        self: "TTensor[_3DShape, T_DType, T_Device]",
        idx: _int,
        /,
    ) -> "TTensor[_2DShape, T_DType, T_Device]": ...

    @overload
    def __getitem__(  # type: ignore
        self: "TTensor[_0DShape, T_DType, T_Device]",
        idx: None,
        /,
    ) -> "TTensor[_1DShape, T_DType, T_Device]": ...

    @overload
    def __getitem__(  # type: ignore
        self: "TTensor[_1DShape, T_DType, T_Device]",
        idx: None,
        /,
    ) -> "TTensor[_2DShape, T_DType, T_Device]": ...

    @overload
    def __getitem__(  # type: ignore
        self: "TTensor[_2DShape, T_DType, T_Device]",
        idx: None,
        /,
    ) -> "TTensor[_3DShape, T_DType, T_Device]": ...

    @overload
    def __getitem__(
        self: "TTensor[T_Shape, T_DType, T_Device]",
        sl: slice,
        /,
    ) -> "TTensor[T_Shape, T_DType, T_Device]": ...

    @overload
    def __getitem__(self, *args) -> "TTensor": ...

    @overload
    def __ne__(  # type: ignore
        self: "TTensor[T_Shape, Any, T_Device]",
        other: "TTensor[T_Shape, Any, T_Device]",
    ) -> "TTensor[T_Shape, Literal[DTypeEnum.bool], T_Device]": ...

    @overload
    def __ne__(  # type: ignore
        self,
        other: Any,
    ) -> "TTensor[_DefaultShape, Literal[DTypeEnum.bool], T_Device]": ...

    @overload
    def abs(  # type: ignore
        self: "TTensor[T_Shape, T_DType, T_Device]",
    ) -> "TTensor[T_Shape, T_DType, T_Device]": ...

    @overload
    def absolute(  # type: ignore
        self: "TTensor[T_Shape, T_DType, T_Device]",
    ) -> "TTensor[T_Shape, T_DType, T_Device]": ...

    @overload
    def acos(  # type: ignore
        self: "TTensor[T_Shape, T_DType, T_Device]",
    ) -> "TTensor[T_Shape, T_DType, T_Device]": ...

    @overload
    def all(self, dim: Literal[None] = None) -> "BoolTensor0D":  # type: ignore
        ...

    @overload
    def all(  # type: ignore
        self,
        dim: Union[_int, Tuple[_int, ...]],
        keepdim: _bool = False,
    ) -> "BoolTensor": ...

    @overload
    def any(  # type: ignore
        self,
        dim: Literal[None] = None,
    ) -> "TTensor[_0DShape, Literal[DTypeEnum.bool], T_Device]": ...

    @overload
    def any(  # type: ignore
        self: "TTensor[T_Shape, Any, T_Device]",
        dim: Union[_int, Tuple[_int, ...]],
        keepdim: _bool = False,
    ) -> "BoolTensor[T_Shape, T_Device]": ...

    @overload
    def bool(  # type: ignore
        self: "TTensor[T_Shape, Any, T_Device]",
    ) -> "BoolTensor[T_Shape, T_Device]": ...

    @overload
    def contiguous(  # type: ignore
        self: "TTensor[T_Shape, T_DType, T_Device]",
    ) -> "TTensor[T_Shape, T_DType, T_Device]": ...

    @overload
    def cpu(  # type: ignore
        self: "TTensor[T_Shape, T_DType, T_Device]",
    ) -> "CPUTensor[T_Shape, T_DType]": ...

    @overload
    def cuda(self: "TTensor") -> "TTensor":  # type: ignore
        ...

    @overload
    def double(self) -> "DoubleTensor":  # type: ignore
        ...

    @overload
    def eq(self: "TTensor", other: Union[torch.Tensor, BuiltinNumber]) -> "BoolTensor":  # type: ignore
        ...

    @overload
    def equal(self: "TTensor", other: torch.Tensor) -> _bool:  # type: ignore
        ...

    @overload
    def float(self) -> "FloatTensor":  # type: ignore
        ...

    @overload
    def half(self) -> "HalfTensor":  # type: ignore
        ...

    @overload
    def int(self) -> "IntTensor":  # type: ignore
        ...

    @overload
    def is_complex(self) -> _bool:  # type: ignore
        ...

    @overload
    def is_floating_point(self) -> _bool:  # type: ignore
        ...

    @overload
    def is_signed(self) -> _bool:  # type: ignore
        ...

    @overload
    def isfinite(self) -> "BoolTensor":  # type: ignore
        ...

    @overload
    def isinf(self) -> "BoolTensor":  # type: ignore
        ...

    @overload
    def isnan(self) -> "BoolTensor":  # type: ignore
        ...

    @overload
    def item(self) -> BuiltinNumber:  # type: ignore
        ...

    @overload
    def long(self) -> "LongTensor":  # type: ignore
        ...

    @overload
    def mean(self, dim: Literal[None] = None) -> "Tensor0D":  # type: ignore
        ...

    @overload
    def mean(self: "Tensor0D", dim: _int) -> "Tensor0D": ...

    @overload
    def mean(self: "Tensor1D", dim: _int) -> "Tensor0D":  # type: ignore
        ...

    @overload
    def mean(self: "Tensor2D", dim: _int) -> "Tensor1D":  # type: ignore
        ...

    @overload
    def mean(self: "Tensor3D", dim: _int) -> "Tensor2D":  # type: ignore
        ...

    @overload
    def mean(self, dim: _int) -> "Tensor":  # type: ignore
        ...

    @overload
    def reshape(  # type: ignore
        self: "TTensor[T_Shape, T_DType, T_Device]",
        size: T_Shape2,
    ) -> "TTensor[T_Shape2, T_DType, T_Device]": ...

    @overload
    def reshape(self, size0: _int) -> "Tensor1D": ...  # type: ignore

    @overload
    def reshape(self, size0: _int, size1: _int) -> "Tensor2D": ...  # type: ignore

    @overload
    def reshape(self, size0: _int, size1: _int, size2: _int) -> "Tensor3D":  # type: ignore
        ...

    @overload
    def short(self) -> "ShortTensor":  # type: ignore
        ...

    @overload
    def size(self: "TTensor[T_Shape, T_DType, T_Device]") -> T_Shape:  # type: ignore
        ...

    @overload
    def squeeze(self, dim: Optional[_int] = None) -> "TTensor":  # type: ignore
        ...

    @overload
    def sum(self, dim: Literal[None] = None) -> "Tensor0D":  # type: ignore
        ...

    @overload
    def sum(self: "Tensor1D", dim: _int) -> "Tensor0D": ...

    @overload
    def sum(self: "Tensor2D", dim: _int) -> "Tensor1D":  # type: ignore
        ...

    @overload
    def sum(self: "Tensor3D", dim: _int) -> "Tensor2D":  # type: ignore
        ...

    @overload
    def sum(self, dim: Optional[_int] = None) -> "Tensor":  # type: ignore
        ...

    @overload
    def tolist(self) -> Any:  # type: ignore
        ...

    @overload
    def unsqueeze(self: "Tensor0D", dim: _int) -> "Tensor1D":  # type: ignore
        ...

    @overload
    def unsqueeze(self: "Tensor1D", dim: _int) -> "Tensor2D":  # type: ignore
        ...

    @overload
    def unsqueeze(self: "Tensor2D", dim: _int) -> "Tensor3D":  # type: ignore
        ...

    @overload
    def unsqueeze(self, dim: _int) -> "Tensor":  # type: ignore
        ...

    @overload
    def view(  # type: ignore
        self: "TTensor[T_Shape, T_DType, T_Device]",
        size: T_Shape2,
    ) -> "TTensor[T_Shape2, T_DType, T_Device]": ...

    @overload
    def view(
        self: "TTensor[T_Shape, T_DType, T_Device]",
        size0: _int,
    ) -> "Tensor1D[T_DType, T_Device]": ...

    @overload
    def view(
        self: "TTensor[T_Shape, T_DType, T_Device]",
        size0: _int,
        size1: _int,
    ) -> "Tensor2D[T_DType, T_Device]": ...

    @overload
    def view(
        self: "TTensor[T_Shape, T_DType, T_Device]",
        size0: _int,
        size1: _int,
        size2: _int,
    ) -> "Tensor3D[T_DType, T_Device]": ...

    @overload
    def view(self, *size: _int) -> "TTensor":  # type: ignore
        ...

    @overload
    def view(self, dtype: torch.dtype) -> "Tensor":  # type: ignore
        ...

    ndim: _ndim_descriptor  # type: ignore
    shape: T_Shape  # type: ignore

    __eq__ = torch.Tensor.__eq__  # noqa: F811  # type: ignore
    __getitem__ = torch.Tensor.__getitem__  # noqa: F811  # type: ignore
    __ne__ = torch.Tensor.__ne__  # noqa: F811  # type: ignore
    abs = torch.Tensor.abs  # noqa: F811  # type: ignore
    absolute = torch.Tensor.absolute  # noqa: F811  # type: ignore
    acos = torch.Tensor.acos  # noqa: F811  # type: ignore
    all = torch.Tensor.all  # noqa: F811  # type: ignore
    any = torch.Tensor.any  # noqa: F811  # type: ignore
    bool = torch.Tensor.bool  # noqa: F811  # type: ignore
    contiguous = torch.Tensor.contiguous  # noqa: F811  # type: ignore
    cpu = torch.Tensor.cpu  # noqa: F811  # type: ignore
    cuda = torch.Tensor.cuda  # noqa: F811  # type: ignore
    double = torch.Tensor.double  # noqa: F811  # type: ignore
    eq = torch.Tensor.eq  # noqa: F811  # type: ignore
    equal = torch.Tensor.equal  # noqa: F811  # type: ignore
    float = torch.Tensor.float  # noqa: F811  # type: ignore
    half = torch.Tensor.half  # noqa: F811  # type: ignore
    is_complex = torch.Tensor.is_complex  # noqa: F811  # type: ignore
    is_floating_point = torch.Tensor.is_floating_point  # noqa: F811  # type: ignore
    is_signed = torch.Tensor.is_signed  # noqa: F811  # type: ignore
    isfinite = torch.Tensor.isfinite  # noqa: F811  # type: ignore
    isinf = torch.Tensor.isinf  # noqa: F811  # type: ignore
    isnan = torch.Tensor.isnan  # noqa: F811  # type: ignore
    int = torch.Tensor.int  # noqa: F811  # type: ignore
    item = torch.Tensor.item  # noqa: F811  # type: ignore
    long = torch.Tensor.long  # noqa: F811  # type: ignore
    mean = torch.Tensor.mean  # noqa: F811  # type: ignore
    reshape = torch.Tensor.reshape  # noqa: F811  # type: ignore
    short = torch.Tensor.short  # noqa: F811  # type: ignore
    size = torch.Tensor.size  # noqa: F811  # type: ignore
    squeeze = torch.Tensor.squeeze  # noqa: F811  # type: ignore
    sum = torch.Tensor.sum  # noqa: F811  # type: ignore
    tolist = torch.Tensor.tolist  # noqa: F811  # type: ignore
    unsqueeze = torch.Tensor.unsqueeze  # noqa: F811  # type: ignore
    view = torch.Tensor.view  # noqa: F811  # type: ignore


class Tensor0D(
    Generic[T_DType, T_Device],
    TTensor[_0DShape, T_DType, T_Device],
): ...


class Tensor1D(
    Generic[T_DType, T_Device],
    TTensor[_1DShape, T_DType, T_Device],
): ...


class Tensor2D(
    Generic[T_DType, T_Device],
    TTensor[_2DShape, T_DType, T_Device],
): ...


class Tensor3D(
    Generic[T_DType, T_Device],
    TTensor[_3DShape, T_DType, T_Device],
): ...


class BoolTensor(
    Generic[T_Shape, T_Device],
    TTensor[T_Shape, DTypeEnum.bool, T_Device],
): ...


class BoolTensor0D(
    Generic[T_Device],
    TTensor[_0DShape, DTypeEnum.bool, T_Device],
): ...


class CPUTensor(
    Generic[T_Shape, T_DType],
    TTensor[T_Shape, T_DType, DeviceEnum.cpu],
): ...


class CUDAFloatTensor2D(
    TTensor[_2DShape, DTypeEnum.f32, DeviceEnum.cuda],
): ...


class DoubleTensor(
    Generic[T_Shape, T_Device],
    TTensor[T_Shape, DTypeEnum.f64, T_Device],
): ...


class FloatTensor(
    Generic[T_Shape, T_Device],
    TTensor[T_Shape, DTypeEnum.f32, T_Device],
): ...


class FloatTensor2D(
    Generic[T_Device],
    TTensor[_2DShape, DTypeEnum.f32, T_Device],
): ...


class HalfTensor(
    Generic[T_Shape, T_Device],
    TTensor[T_Shape, DTypeEnum.half, T_Device],
): ...


class IntTensor(
    Generic[T_Shape, T_Device],
    TTensor[T_Shape, DTypeEnum.int, T_Device],
): ...


class LongTensor(
    Generic[T_Shape, T_Device],
    TTensor[T_Shape, DTypeEnum.long, T_Device],
): ...


class ShortTensor(
    Generic[T_Shape, T_Device],
    TTensor[T_Shape, DTypeEnum.short, T_Device],
): ...


# TODO: rm
x = Tensor2D[DTypeEnum.f32, DeviceEnum.cuda]([[]])  # type: ignore
y = x.view((1, 2, 0))
s = y.shape
n = y.ndim
m = x.ndim
z = x[0]
a = z[0]
