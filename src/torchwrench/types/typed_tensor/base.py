#!/usr/bin/env python
# -*- coding: utf-8 -*-

from builtins import bool as _bool
from builtins import int as _int
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
    overload,
)
from typing import Literal as L

import torch
from pythonwrench.typing.classes import BuiltinNumber
from torch._C import _TensorMeta
from typing_extensions import Self, Type, TypeAlias, TypeVar

from torchwrench.core.device_enum import DeviceEnum
from torchwrench.core.dtype_enum import DTypeEnum
from torchwrench.core.make import DeviceLike, DTypeLike, as_device, as_dtype

_0DShape = Tuple[()]
_1DShape = Tuple[_int]
_2DShape = Tuple[_int, _int]
_3DShape = Tuple[_int, _int, _int]
_4DShape = Tuple[_int, _int, _int, _int]


_AnyShape: TypeAlias = Any
_AnyDType: TypeAlias = Any
_AnyDevice: TypeAlias = Any

_ShapeGenericType: TypeAlias = Union[Tuple[_int, ...], _AnyShape]
_DTypeGenericType: TypeAlias = Union[DTypeEnum, _AnyDType]
_DeviceGenericType: TypeAlias = Union[DeviceEnum, _AnyDevice]


T_Shape = TypeVar(
    "T_Shape",
    bound=_ShapeGenericType,
    default=_AnyShape,
    covariant=True,
)
T_DType = TypeVar(
    "T_DType",
    bound=_DTypeGenericType,
    default=_AnyDType,
    covariant=True,
)
T_Device = TypeVar(
    "T_Device",
    bound=_DeviceGenericType,
    default=_AnyDevice,
    covariant=True,
)

T_Shape2 = TypeVar(
    "T_Shape2",
    bound=_ShapeGenericType,
    default=_AnyShape,
    covariant=True,
)
T_DType2 = TypeVar(
    "T_DType2",
    bound=_DTypeGenericType,
    default=_AnyDType,
    covariant=True,
)
T_Device2 = TypeVar(
    "T_Device2",
    bound=_DeviceGenericType,
    default=_AnyDevice,
    covariant=True,
)

T_Axis0 = TypeVar(
    "T_Axis0",
    bound=_int,
    default=_int,
    covariant=True,
)
T_Axis1 = TypeVar(
    "T_Axis1",
    bound=_int,
    default=_int,
    covariant=True,
)
T_Axis2 = TypeVar(
    "T_Axis2",
    bound=_int,
    default=_int,
    covariant=True,
)


@dataclass
class _TTensorGenerics:
    shape: Union[Tuple[Optional[_int], ...], None]
    dtype: Union[torch.dtype, None]
    device: Union[torch.device, None]

    def is_compatible_with_tensor(self, x: torch.Tensor) -> _bool:
        return (
            self.is_compatible_with_shape(x.shape)
            and self.is_compatible_with_dtype(x.dtype)
            and self.is_compatible_with_device(x.device)
        )

    def is_compatible_with_tensor_generics(self, x: Self) -> _bool:
        return (
            (x.shape is None or self.is_compatible_with_shape(x.shape))
            and (x.dtype is None or self.is_compatible_with_dtype(x.dtype))
            and (x.device is None or self.is_compatible_with_device(x.device))
        )

    def is_compatible_with_shape(self, shape: Tuple[Optional[_int], ...]) -> _bool:
        if self.shape is None:
            return True

        if len(get_args(self.shape)) != len(shape):
            return False

        valid = [
            (s1 is None) or (s2 is None) or (s1 == s2)
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

    def is_compatible_with_device(
        self,
        device: Union[torch.device, str, _int],
    ) -> _bool:
        if self.device is None:
            return True

        return as_device(self.device) == as_device(device)

    @property
    def ndim(self) -> Optional[_int]:
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


def _generic_shape_to_shape(
    generic_shape: _ShapeGenericType,
) -> Optional[Tuple[Optional[_int], ...]]:
    if isinstance(generic_shape, TypeVar) or generic_shape is _AnyShape:
        return None

    return tuple(
        get_args(s)[0] if (get_origin(s) is L) else None
        for s in get_args(generic_shape)
    )


def _generic_dtype_to_dtype(generic_dtype: _DTypeGenericType) -> Optional[torch.dtype]:
    if isinstance(generic_dtype, TypeVar) or generic_dtype is _AnyDType:
        return None

    if isinstance(generic_dtype, DTypeEnum):
        return generic_dtype.dtype

    msg = f"Invalid argument {generic_dtype=} (expected DTypeEnum or None)."
    raise TypeError(msg)


def _generic_device_to_device(
    generic_device: _DeviceGenericType,
) -> Optional[torch.device]:
    if isinstance(generic_device, TypeVar) or generic_device is _AnyDevice:
        return None
    else:
        return as_device(generic_device)


def _cls_to_generics(cls: type) -> _TTensorGenerics:
    """Get the generic parameters of a TTensor subclass."""

    if cls is TTensor or get_origin(cls) is TTensor:
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

    if len(args) < 3:
        missing = (_AnyShape, _AnyDType, _AnyDevice)[len(args) :]
        args = args + missing

    if len(args) != 3:
        msg = (
            f"Expected 3 generic parameters for TTensor, but got {len(args)} in {cls}."
        )
        raise TypeError(msg)

    generic_shape, generic_dtype, generic_device = args
    shape = _generic_shape_to_shape(generic_shape)
    dtype = _generic_dtype_to_dtype(generic_dtype)
    device = _generic_device_to_device(generic_device)

    return _TTensorGenerics(
        shape=shape,
        dtype=dtype,
        device=device,
    )


def _get_generics(cls_or_instance: Any) -> _TTensorGenerics:
    """Get the generic parameters of a TTensor subclass or instance."""
    if not isinstance(cls_or_instance, type):
        cls_or_instance = type(cls_or_instance)
    return _cls_to_generics(cls_or_instance)


class _TTensorMeta(_TensorMeta):
    def __instancecheck__(self, instance: Any) -> _bool:
        """Called method to check isinstance(instance, TTensor)"""
        if not isinstance(instance, torch.Tensor):
            return False

        gen = _cls_to_generics(self)
        return gen.is_compatible_with_tensor(instance)

    def __subclasscheck__(self, subclass: Any) -> _bool:
        """Called method to check issubclass(subclass, TTensor)"""
        self_generics = _cls_to_generics(self)
        other_generics = _cls_to_generics(subclass)
        return self_generics.is_compatible_with_tensor_generics(other_generics)


class _ndim_descriptor:
    @overload
    def __get__(
        self,
        instance: "TTensor[_0DShape, Any, Any]",
        owner: Any,
    ) -> L[0]: ...
    @overload
    def __get__(
        self,
        instance: "TTensor[_1DShape, Any, Any]",
        owner: Any,
    ) -> L[1]: ...
    @overload
    def __get__(
        self,
        instance: "TTensor[_2DShape, Any, Any]",
        owner: Any,
    ) -> L[2]: ...
    @overload
    def __get__(
        self,
        instance: "TTensor[_3DShape, Any, Any]",
        owner: Any,
    ) -> L[3]: ...
    @overload
    def __get__(
        self,
        instance: "TTensor[_4DShape, Any, Any]",
        owner: Any,
    ) -> L[4]: ...
    @overload
    def __get__(
        self,
        instance: "TTensor[Any, Any, Any]",
        owner: Any,
    ) -> _int: ...

    def __get__(self, instance: object, owner: Any) -> _int:
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
        cls_device = gen.device

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

        # Sanity checks for device
        if device is None:
            if cls_device is not None:
                device = cls_device
        elif cls_device is None or device == cls_device:
            pass
        else:
            msg = f"Invalid argument {device=} for {cls.__name__}. (expected {cls_device})"
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
    def __eq__(  # type: ignore
        self,
        other: "TTensor[T_Shape, T_DType2, T_Device]",
    ) -> "TTensor[T_Shape, L[DTypeEnum.bool], T_Device]": ...

    @overload
    def __eq__(  # type: ignore
        self,
        other: BuiltinNumber,
    ) -> "TTensor[T_Shape, L[DTypeEnum.bool], T_Device]": ...

    @overload
    def __eq__(  # type: ignore
        self,
        other: Any,
    ) -> "TTensor[_AnyShape, L[DTypeEnum.bool], T_Device]": ...

    @overload
    def __ge__(  # type: ignore
        self,
        other: Any,
    ) -> "TTensor[_AnyShape, L[DTypeEnum.bool], T_Device]": ...

    @overload
    def __getitem__(  # type: ignore
        self: "TTensor[_1DShape, T_DType, T_Device]",
        idx: _int,
        /,
    ) -> "TTensor[_0DShape, T_DType, T_Device]": ...

    @overload
    def __getitem__(  # type: ignore
        self: "TTensor[Tuple[T_Axis0, T_Axis1], T_DType, T_Device]",
        idx: _int,
        /,
    ) -> "TTensor[Tuple[T_Axis1], T_DType, T_Device]": ...

    @overload
    def __getitem__(  # type: ignore
        self: "TTensor[Tuple[T_Axis0, T_Axis1, T_Axis2], T_DType, T_Device]",
        idx: _int,
        /,
    ) -> "TTensor[Tuple[T_Axis1, T_Axis2], T_DType, T_Device]": ...

    @overload
    def __getitem__(  # type: ignore
        self: "TTensor[_0DShape, T_DType, T_Device]",
        idx: None,
        /,
    ) -> "TTensor[_1DShape, T_DType, T_Device]": ...

    @overload
    def __getitem__(  # type: ignore
        self: "TTensor[Tuple[T_Axis0], T_DType, T_Device]",
        idx: None,
        /,
    ) -> "TTensor[Tuple[L[1], T_Axis0], T_DType, T_Device]": ...

    @overload
    def __getitem__(  # type: ignore
        self: "TTensor[Tuple[T_Axis0, T_Axis1], T_DType, T_Device]",
        idx: None,
        /,
    ) -> "TTensor[Tuple[L[1], T_Axis0, T_Axis1], T_DType, T_Device]": ...

    @overload
    def __getitem__(  # type: ignore
        self: "TTensor[T_Shape, T_DType, T_Device]",
        sl: slice,
        /,
    ) -> "TTensor[T_Shape, T_DType, T_Device]": ...

    @overload
    def __getitem__(
        self: "TTensor[T_Shape, T_DType, T_Device]",
        *args,
    ) -> "TTensor[_AnyShape, T_DType, T_Device]": ...

    @overload
    def __gt__(  # type: ignore
        self: "TTensor[T_Shape, T_DType, T_Device]", other: Any
    ) -> "TTensor[_AnyShape, L[DTypeEnum.bool], T_Device]": ...

    @overload
    def __le__(  # type: ignore
        self: "TTensor[T_Shape, T_DType, T_Device]", other: Any
    ) -> "TTensor[_AnyShape, L[DTypeEnum.bool], T_Device]": ...

    @overload
    def __lt__(  # type: ignore
        self: "TTensor[T_Shape, T_DType, T_Device]", other: Any
    ) -> "TTensor[_AnyShape, L[DTypeEnum.bool], T_Device]": ...

    @overload
    def __ne__(  # type: ignore
        self: "TTensor[T_Shape, Any, T_Device]",
        other: "TTensor[T_Shape, Any, T_Device]",
    ) -> "TTensor[T_Shape, L[DTypeEnum.bool], T_Device]": ...

    @overload
    def __ne__(  # type: ignore
        self,
        other: Any,
    ) -> "TTensor[_AnyShape, L[DTypeEnum.bool], T_Device]": ...

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
        self,
    ) -> "TTensor[T_Shape, T_DType, T_Device]": ...

    @overload
    def all(  # type: ignore
        self,
        dim: L[None] = None,
    ) -> "TTensor[_0DShape, L[DTypeEnum.bool], T_Device]": ...

    @overload
    def all(  # type: ignore
        self,
        dim: Union[_int, Tuple[_int, ...]],
        keepdim: _bool = False,
    ) -> "TTensor[_AnyShape, L[DTypeEnum.bool], T_Device]": ...

    @overload
    def any(  # type: ignore
        self,
        dim: L[None] = None,
    ) -> "TTensor[_0DShape, L[DTypeEnum.bool], T_Device]": ...

    @overload
    def any(  # type: ignore
        self,
        dim: Union[_int, Tuple[_int, ...]],
        keepdim: _bool = False,
    ) -> "TTensor[_AnyShape, L[DTypeEnum.bool], T_Device]": ...

    @overload
    def bool(  # type: ignore
        self,
    ) -> "TTensor[T_Shape, L[DTypeEnum.bool], T_Device]": ...

    @overload
    def contiguous(  # type: ignore
        self: "TTensor[T_Shape, T_DType, T_Device]",
    ) -> "TTensor[T_Shape, T_DType, T_Device]": ...

    @overload
    def cpu(  # type: ignore
        self: "TTensor[T_Shape, T_DType, T_Device]",
    ) -> "TTensor[T_Shape, T_DType, L[DeviceEnum.cpu]]": ...

    @overload
    def cuda(  # type: ignore
        self,
    ) -> "TTensor[T_Shape, T_DType, L[DeviceEnum.cuda]]": ...

    @overload
    def double(self) -> "TTensor[T_Shape, L[DTypeEnum.double], T_Device]":  # type: ignore
        ...

    @overload
    def eq(  # type: ignore
        self: "TTensor",
        other: Union[torch.Tensor, BuiltinNumber],
    ) -> "TTensor[_AnyShape, L[DTypeEnum.bool], T_Device]": ...

    @overload
    def equal(self: "TTensor", other: torch.Tensor) -> _bool:  # type: ignore
        ...

    @overload
    def float(self) -> "TTensor[T_Shape, L[DTypeEnum.float], T_Device]":  # type: ignore
        ...

    @overload
    def half(self) -> "TTensor[T_Shape, L[DTypeEnum.half], T_Device]":  # type: ignore
        ...

    @overload
    def int(self) -> "TTensor[T_Shape, L[DTypeEnum.int], T_Device]":  # type: ignore
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

    def isfinite(self) -> "TTensor[T_Shape, L[DTypeEnum.bool], T_Device]": ...

    def isinf(self) -> "TTensor[T_Shape, L[DTypeEnum.bool], T_Device]": ...

    def isnan(self) -> "TTensor[T_Shape, L[DTypeEnum.bool], T_Device]": ...

    @overload
    def item(self) -> BuiltinNumber:  # type: ignore
        ...

    @overload
    def long(self) -> "TTensor[T_Shape, L[DTypeEnum.long], T_Device]":  # type: ignore
        ...

    @overload
    def mean(self, dim: L[None] = None) -> "TTensor[_0DShape, T_DType, T_Device]":  # type: ignore
        ...

    @overload
    def mean(
        self: "TTensor[_1DShape, T_DType, T_Device]", dim: _int
    ) -> "TTensor[_0DShape, T_DType, T_Device]":  # type: ignore
        ...

    @overload
    def mean(
        self: "TTensor[_2DShape, T_DType, T_Device]", dim: _int
    ) -> "TTensor[_1DShape, T_DType, T_Device]":  # type: ignore
        ...

    @overload
    def mean(
        self: "TTensor[_3DShape, T_DType, T_Device]", dim: _int
    ) -> "TTensor[_2DShape, T_DType, T_Device]":  # type: ignore
        ...

    @overload
    def mean(self, dim: _int) -> "TTensor[_AnyShape, T_DType, T_Device]":  # type: ignore
        ...

    @overload
    def reshape(  # type: ignore
        self: "TTensor[T_Shape, T_DType, T_Device]",
        size: T_Shape2,
    ) -> "TTensor[T_Shape2, T_DType, T_Device]": ...

    @overload
    def reshape(self, size0: _int) -> "TTensor[_1DShape, T_DType, T_Device]": ...

    @overload
    def reshape(
        self, size0: _int, size1: _int
    ) -> "TTensor[_2DShape, T_DType, T_Device]": ...  # type: ignore

    @overload
    def reshape(  # type: ignore
        self,
        size0: _int,
        size1: _int,
        size2: _int,
    ) -> "TTensor[_3DShape, T_DType, T_Device]":  # type: ignore
        ...

    @overload
    def short(self) -> "TTensor[T_Shape, DTypeEnum.short, T_Device]":  # type: ignore
        ...

    @overload
    def size(self: "TTensor[T_Shape, T_DType, T_Device]") -> T_Shape:  # type: ignore
        ...

    @overload
    def squeeze(  # type: ignore
        self: "TTensor[T_Shape, T_DType, T_Device]",
        dim: Optional[_int] = None,
    ) -> "TTensor[_AnyShape, T_DType, T_Device]": ...

    @overload
    def sum(self, dim: L[None] = None) -> "TTensor[_0DShape, T_DType, T_Device]":  # type: ignore
        ...

    @overload
    def sum(
        self: "TTensor[_1DShape, T_DType, T_Device]", dim: _int
    ) -> "TTensor[_0DShape, T_DType, T_Device]": ...

    @overload
    def sum(
        self: "TTensor[_2DShape, T_DType, T_Device]", dim: _int
    ) -> "TTensor[_1DShape, T_DType, T_Device]":  # type: ignore
        ...

    @overload
    def sum(
        self: "TTensor[_3DShape, T_DType, T_Device]", dim: _int
    ) -> "TTensor[_2DShape, T_DType, T_Device]":  # type: ignore
        ...

    @overload
    def sum(self, dim: Optional[_int] = None) -> "TTensor[_0DShape, T_DType, T_Device]":  # type: ignore
        ...

    @overload
    def tolist(self) -> Any:  # type: ignore
        ...

    @overload
    def unsqueeze(  # type: ignore
        self: "TTensor[_0DShape, T_DType, T_Device]", dim: _int
    ) -> "TTensor[_1DShape, T_DType, T_Device]": ...

    @overload
    def unsqueeze(  # type: ignore
        self: "TTensor[_1DShape, T_DType, T_Device]", dim: _int
    ) -> "TTensor[_2DShape, T_DType, T_Device]": ...

    @overload
    def unsqueeze(  # type: ignore
        self: "TTensor[_2DShape, T_DType, T_Device]", dim: _int
    ) -> "TTensor[_3DShape, T_DType, T_Device]": ...

    @overload
    def unsqueeze(  # type: ignore
        self: "TTensor[_3DShape, T_DType, T_Device]", dim: _int
    ) -> "TTensor[_4DShape, T_DType, T_Device]": ...

    @overload
    def unsqueeze(  # type: ignore
        self: "TTensor[Any, T_DType, T_Device]", dim: _int
    ) -> "TTensor[Any, T_DType, T_Device]": ...

    @overload
    def view(  # type: ignore
        self: "TTensor[T_Shape, T_DType, T_Device]",
        size: T_Shape2,
    ) -> "TTensor[T_Shape2, T_DType, T_Device]": ...

    @overload
    def view(
        self: "TTensor[T_Shape, T_DType, T_Device]",
        size0: _int,
    ) -> "TTensor[_1DShape, T_DType, T_Device]": ...

    @overload
    def view(
        self: "TTensor[T_Shape, T_DType, T_Device]",
        size0: _int,
        size1: _int,
    ) -> "TTensor[_2DShape, T_DType, T_Device]": ...

    @overload
    def view(
        self: "TTensor[T_Shape, T_DType, T_Device]",
        size0: _int,
        size1: _int,
        size2: _int,
    ) -> "TTensor[_3DShape, T_DType, T_Device]": ...

    @overload
    def view(self, *size: _int) -> "TTensor[_AnyShape, T_DType, T_Device]":  # type: ignore
        ...

    @overload
    def view(self, dtype: torch.dtype) -> "TTensor[T_Shape, _AnyDType, T_Device]":  # type: ignore
        ...

    ndim: _ndim_descriptor  # type: ignore
    shape: T_Shape  # type: ignore

    __eq__ = torch.Tensor.__eq__  # noqa: F811  # type: ignore
    __ge__ = torch.Tensor.__ge__  # noqa: F811  # type: ignore
    __getitem__ = torch.Tensor.__getitem__  # noqa: F811  # type: ignore
    __gt__ = torch.Tensor.__gt__  # noqa: F811  # type: ignore
    __le__ = torch.Tensor.__le__  # noqa: F811  # type: ignore
    __lt__ = torch.Tensor.__lt__  # noqa: F811  # type: ignore
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
