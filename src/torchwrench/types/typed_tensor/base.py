#!/usr/bin/env python
# -*- coding: utf-8 -*-

from builtins import bool as _bool
from builtins import complex as _complex
from builtins import float as _float
from builtins import int as _int
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    List,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
    overload,
)
from typing import Literal as L

import torch
from pythonwrench.typing.classes import BuiltinNumber, EllipsisType
from torch._C import _TensorMeta
from typing_extensions import Self, Type, TypeAlias, TypeVar

from torchwrench.core import dtype_enum_v2 as dtypes
from torchwrench.core.device_enum import (
    CPUDeviceType,
    CUDADeviceType,
    DeviceBase,
    device_cls_to_torch_device,
)
from torchwrench.core.dtype_enum_v2 import (
    DTypeBase,
    dtype_cls_to_dtype,
    dtype_to_dtype_cls,
)
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
_DTypeGenericType: TypeAlias = Union[DTypeBase, _AnyDType]
_DeviceGenericType: TypeAlias = Union[DeviceBase, _AnyDevice]


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
    default=CPUDeviceType,
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

    def is_compatible_with_dtype(self, dtype: Union[torch.dtype, DTypeBase]) -> _bool:
        """Check if the dtype is compatible with the generic dtype."""
        if self.dtype is None or dtype is None:
            return True

        if isinstance(dtype, torch.dtype):
            dtype_enum = dtype_to_dtype_cls(dtype)
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

    if isinstance(generic_dtype, type) and DTypeBase in generic_dtype.__mro__:
        return dtype_cls_to_dtype(generic_dtype)

    msg = f"Invalid argument {generic_dtype=} (expected DTypeEnum or None)."
    raise TypeError(msg)


def _generic_device_to_device(
    generic_device: _DeviceGenericType,
) -> Optional[torch.device]:
    if isinstance(generic_device, TypeVar) or generic_device is _AnyDevice:
        return None
    else:
        return device_cls_to_torch_device(generic_device)


def _cls_to_generics(cls: type) -> _TTensorGenerics:
    """Get the generic parameters of a TTensor subclass."""

    if (
        hasattr(cls, "__gen_shape__")
        and hasattr(cls, "__gen_dtype__")
        and hasattr(cls, "__gen_device__")
    ):
        args = (cls.__gen_shape__, cls.__gen_dtype__, cls.__gen_device__)
    elif cls is TTensor or get_origin(cls) is TTensor:
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
    # TODO: rm
    # print(f"{cls=}")
    # print(f"{generic_shape=}")
    # print(f"{generic_dtype=}")
    # print(f"{generic_device=}")
    # breakpoint()

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


def _resolve_ttensors_generics(cls, args):
    # Find TTensor[...] base
    for base in getattr(cls, "__orig_bases__", ()):
        if get_origin(base) is TTensor:
            base_args = list(get_args(base))  # (_2DShape, T_DType, T_Device)
            break
    else:
        base_args = [None, None, None]

    # Replace TypeVars with provided args
    resolved = []
    arg_iter = iter(args)

    for a in base_args:
        if isinstance(a, TypeVar):
            resolved.append(next(arg_iter))
        else:
            resolved.append(a)

    return tuple(resolved)


class TTensor(
    Generic[T_Shape, T_DType, T_Device],
    torch.Tensor,
    metaclass=_TTensorMeta,
):
    _DEFAULT_DTYPE: Optional[Type[DTypeBase]] = None

    @classmethod
    def __class_getitem__(cls, gen_args):
        # # Create a specialized subclass: Box[int], Box[str], etc.
        # name = f"{cls.__name__}[" + ",".join(map(str, args)) + "]"
        # print(f"{cls=}")
        # print(f"{args=}")
        # return type(name, (cls,), {"__orig_bases__": [TTensor], "__custom__": args})    # Create a specialized subclass with proper generic parameters

        # # Create a specialized subclass with proper generic parameters
        name = f"{cls.__name__}[" + ",".join(map(str, gen_args)) + "]"

        # if cls is TTensor:
        #     generic_args = args[0]
        # else:
        #     generic_args = ()
        # breakpoint()

        # TODO: rm
        # print(f"{generic_args=}")
        # breakpoint()

        # # Get the current class's __orig_bases__ to extract the shape constraint
        # if hasattr(cls, "__orig_bases__"):
        #     orig_bases = cls.__orig_bases__  # type: ignore
        #     # Find TTensor base
        #     ttensor_base = None
        #     for base in orig_bases:
        #         if get_origin(base) is TTensor:
        #             ttensor_base = base
        #             break
        #     base_args = get_args(ttensor_base) if ttensor_base else ()
        # else:
        #     base_args = ()

        # Extract shape from parent, merge with new args
        # shape_arg = base_args[0] if len(base_args) > 0 else _AnyShape
        # dtype_arg = args[0] if len(args) > 0 else _AnyDType
        # device_arg = args[1] if len(args) > 1 else _AnyDevice

        # Create a wrapper class that stores generics and delegates __new__ to parent
        # class SpecializedTensor:  # type: ignore
        #     __new__ = cls.__new__
        #     # __generic_params__ = (shape_arg, dtype_arg, device_arg)
        #     __generic_params__ = args[0]

        # SpecializedTensor.__name__ = name
        # SpecializedTensor.__qualname__ = name

        # print(f"{SpecializedTensor.__generic_params__=}")
        # breakpoint()
        # return SpecializedTensor  # type: ignore

        # full_generics = _resolve_ttensors_generics(cls, args)
        print("Class getitem:")
        print(f"{cls=}")
        print(f"{gen_args=}")
        # print(f"{full_generics=}")

        # class SpecializedTensor(cls):
        #     __new__ = cls.__new__
        #     __generic_params__ = generic_args

        # SpecializedTensor.__name__ = name
        # SpecializedTensor.__qualname__ = name

        base_cls = cls
        cls = type(name, (cls,), {"__orig__": TTensor})

        if base_cls is TTensor:
            if len(gen_args) > 0:
                gen_shape = gen_args[0]
            else:
                gen_shape = T_Shape

            if len(gen_args) > 1:
                gen_dtype = gen_args[1]
            else:
                gen_dtype = T_DType

            if len(gen_args) > 2:
                gen_device = gen_args[2]
            else:
                gen_device = T_Device

            if not hasattr(cls, "__gen_shape__") or isinstance(
                cls.__gen_shape__, TypeVar
            ):
                cls.__gen_shape__ = gen_shape  # type: ignore

            if not hasattr(cls, "__gen_dtype__") or isinstance(
                cls.__gen_dtype__, TypeVar
            ):
                cls.__gen_dtype__ = gen_dtype  # type: ignore

            if not hasattr(cls, "__gen_device__") or isinstance(
                cls.__gen_device__, TypeVar
            ):
                cls.__gen_device__ = gen_device  # type: ignore

            return super().__class_getitem__((gen_args,))  # type: ignore

        else:
            for arg in gen_args:
                if (
                    not hasattr(cls, "__gen_shape__")
                    or isinstance(cls.__gen_shape__, TypeVar)
                ) and get_origin(arg) is tuple:
                    cls.__gen_shape__ = arg
                if (
                    (
                        not hasattr(cls, "__gen_dtype__")
                        or isinstance(cls.__gen_dtype__, TypeVar)
                    )
                    and isinstance(arg, type)
                    and issubclass(arg, DTypeBase)
                ):
                    cls.__gen_dtype__ = arg
                if (
                    (
                        not hasattr(cls, "__gen_device__")
                        or isinstance(cls.__gen_device__, TypeVar)
                    )
                    and isinstance(arg, type)
                    and issubclass(arg, DeviceBase)
                ):
                    cls.__gen_device__ = arg

        print(f"{cls.__gen_shape__=}")
        print(f"{cls.__gen_dtype__=}")
        print(f"{cls.__gen_device__=}")

        # breakpoint()
        return super().__class_getitem__((gen_args,))  # type: ignore

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

        # TODO: rm
        print("New tensor:")
        print(f"{cls=}")
        print(f"{gen=}")
        print(f"{getattr(cls, '__gen_shape__', None)=}")
        print(f"{getattr(cls, '__gen_dtype__', None)=}")
        print(f"{getattr(cls, '__gen_device__', None)=}")
        # breakpoint()

        # Sanity checks for dtype
        if dtype is None:
            if cls_dtype is not None:
                dtype = cls_dtype
            elif getattr(cls, "_DEFAULT_DTYPE", None) is not None:
                dtype = dtype_cls_to_dtype(cls._DEFAULT_DTYPE)

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
    ) -> "TTensor[T_Shape, dtypes.BoolDType, T_Device]": ...

    @overload
    def __eq__(  # type: ignore
        self,
        other: BuiltinNumber,
    ) -> "TTensor[T_Shape, dtypes.BoolDType, T_Device]": ...

    @overload
    def __eq__(  # type: ignore
        self,
        other: Any,
    ) -> "TTensor[_AnyShape, dtypes.BoolDType, T_Device]": ...

    @overload
    def __ge__(  # type: ignore
        self,
        other: Any,
    ) -> "TTensor[_AnyShape, dtypes.BoolDType, T_Device]": ...

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
    ) -> "TTensor[_AnyShape, dtypes.BoolDType, T_Device]": ...

    @overload
    def __le__(  # type: ignore
        self: "TTensor[T_Shape, T_DType, T_Device]", other: Any
    ) -> "TTensor[_AnyShape, dtypes.BoolDType, T_Device]": ...

    @overload
    def __lt__(  # type: ignore
        self: "TTensor[T_Shape, T_DType, T_Device]", other: Any
    ) -> "TTensor[_AnyShape, dtypes.BoolDType, T_Device]": ...

    @overload
    def __ne__(  # type: ignore
        self: "TTensor[T_Shape, Any, T_Device]",
        other: "TTensor[T_Shape, Any, T_Device]",
    ) -> "TTensor[T_Shape, dtypes.BoolDType, T_Device]": ...

    @overload
    def __ne__(  # type: ignore
        self,
        other: Any,
    ) -> "TTensor[_AnyShape, dtypes.BoolDType, T_Device]": ...

    def abs(
        self: "TTensor[T_Shape, T_DType, T_Device]",
    ) -> "TTensor[T_Shape, T_DType, T_Device]": ...

    def absolute(
        self: "TTensor[T_Shape, T_DType, T_Device]",
    ) -> "TTensor[T_Shape, T_DType, T_Device]": ...

    def acos(
        self,
    ) -> "TTensor[T_Shape, T_DType, T_Device]": ...

    @overload
    def all(  # type: ignore
        self,
        dim: L[None] = None,
    ) -> "TTensor[_0DShape, dtypes.BoolDType, T_Device]": ...

    @overload
    def all(  # type: ignore
        self,
        dim: Union[_int, Tuple[_int, ...]],
        keepdim: _bool = False,
    ) -> "TTensor[_AnyShape, dtypes.BoolDType, T_Device]": ...

    @overload
    def any(  # type: ignore
        self,
        dim: L[None] = None,
    ) -> "TTensor[_0DShape, dtypes.BoolDType, T_Device]": ...

    @overload
    def any(  # type: ignore
        self,
        dim: Union[_int, Tuple[_int, ...]],
        keepdim: _bool = False,
    ) -> "TTensor[_AnyShape, dtypes.BoolDType, T_Device]": ...

    def bool(
        self,
    ) -> "TTensor[T_Shape, dtypes.BoolDType, T_Device]": ...

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> "TTensor[T_Shape, T_DType, T_Device]": ...

    def cpu(
        self,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> "TTensor[T_Shape, T_DType, CPUDeviceType]": ...

    def cuda(
        self,
        device: Union[torch.device, _int, str, None] = None,
        non_blocking: _bool = False,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> "TTensor[T_Shape, T_DType, CUDADeviceType]": ...

    def double(self) -> "TTensor[T_Shape, dtypes.DoubleDType, T_Device]": ...

    def eq(
        self,
        other: Union[torch.Tensor, BuiltinNumber],
    ) -> "TTensor[_AnyShape, dtypes.BoolDType, T_Device]": ...

    def equal(self, other: torch.Tensor) -> _bool: ...

    def float(self) -> "TTensor[T_Shape, dtypes.FloatDType, T_Device]": ...

    def half(self) -> "TTensor[T_Shape, dtypes.HalfDType, T_Device]": ...

    def int(self) -> "TTensor[T_Shape, dtypes.IntDType, T_Device]": ...

    @overload
    def is_complex(self: "TTensor[Any, DTypeBase[L[True], Any, Any], Any]") -> L[True]:  # type: ignore
        ...

    @overload
    def is_complex(
        self: "TTensor[Any, DTypeBase[L[False], Any, Any], Any]",
    ) -> L[False]: ...

    @overload
    def is_complex(self) -> _bool: ...

    @overload
    def is_floating_point(  # type: ignore
        self: "TTensor[Any, DTypeBase[Any, L[True], Any], Any]",
    ) -> L[True]: ...

    @overload
    def is_floating_point(
        self: "TTensor[Any, DTypeBase[Any, L[False], Any], Any]",
    ) -> L[False]: ...

    @overload
    def is_floating_point(self) -> _bool: ...

    @overload
    def is_signed(self: "TTensor[Any, DTypeBase[Any, Any, L[True]], Any]") -> L[True]:  # type: ignore
        ...

    @overload
    def is_signed(
        self: "TTensor[Any, DTypeBase[Any, Any, L[False]], Any]",
    ) -> L[False]: ...

    @overload
    def is_signed(self) -> _bool: ...

    def isfinite(self) -> "TTensor[T_Shape, dtypes.BoolDType, T_Device]": ...

    def isinf(self) -> "TTensor[T_Shape, dtypes.BoolDType, T_Device]": ...

    def isnan(self) -> "TTensor[T_Shape, dtypes.BoolDType, T_Device]": ...

    @overload
    def item(self: "TTensor[_AnyShape, dtypes.BoolDType, _AnyDevice]") -> _bool: ...  # type: ignore

    @overload
    def item(
        self: "TTensor[_AnyShape, DTypeBase[L[False], L[False], Any], _AnyDevice]",
    ) -> _int: ...

    @overload
    def item(
        self: "TTensor[_AnyShape, DTypeBase[L[False], L[True], L[True]], _AnyDevice]",
    ) -> _float: ...

    @overload
    def item(
        self: "TTensor[_AnyShape, DTypeBase[L[True], L[False], L[False]], _AnyDevice]",
    ) -> _complex: ...

    @overload
    def item(self) -> BuiltinNumber: ...

    @overload
    def long(self) -> "TTensor[T_Shape, dtypes.LongDType, T_Device]":  # type: ignore
        ...

    @overload
    def mean(self, dim: L[None] = None) -> "TTensor[_0DShape, T_DType, T_Device]":  # type: ignore
        ...

    @overload
    def mean(
        self: "TTensor[_1DShape, T_DType, T_Device]",
        dim: _int,
    ) -> "TTensor[_0DShape, T_DType, T_Device]": ...

    @overload
    def mean(
        self: "TTensor[_2DShape, T_DType, T_Device]",
        dim: _int,
    ) -> "TTensor[_1DShape, T_DType, T_Device]": ...

    @overload
    def mean(
        self: "TTensor[_3DShape, T_DType, T_Device]",
        dim: _int,
    ) -> "TTensor[_2DShape, T_DType, T_Device]": ...

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

    def short(self) -> "TTensor[T_Shape, dtypes.ShortDType, T_Device]": ...

    @overload
    def size(self, dim: None = None) -> T_Shape:  # type: ignore
        ...
    @overload
    def size(self, dim: _int) -> _int: ...
    @overload
    def size(self, dim: Optional[_int]) -> Any:  # type: ignore
        ...

    @overload
    def squeeze(  # type: ignore
        self,
        dim: _int,
    ) -> "TTensor[_AnyShape, T_DType, T_Device]": ...

    @overload
    def squeeze(
        self,
        dim: torch.Size,
    ) -> "TTensor[_AnyShape, T_DType, T_Device]": ...

    @overload
    def squeeze(
        self,
        *dim: _int,
    ) -> "TTensor[_AnyShape, T_DType, T_Device]": ...

    @overload
    def squeeze(  # type: ignore
        self,
        dim: Union[None, str, EllipsisType] = None,
    ) -> "TTensor[_AnyShape, T_DType, T_Device]": ...

    @overload
    def sum(self, dim: L[None] = None) -> "TTensor[_0DShape, T_DType, T_Device]":  # type: ignore
        ...

    @overload
    def sum(
        self: "TTensor[_1DShape, T_DType, T_Device]",
        dim: _int,
    ) -> "TTensor[_0DShape, T_DType, T_Device]": ...

    @overload
    def sum(
        self: "TTensor[_2DShape, T_DType, T_Device]",
        dim: _int,
    ) -> "TTensor[_1DShape, T_DType, T_Device]": ...

    @overload
    def sum(
        self: "TTensor[_3DShape, T_DType, T_Device]",
        dim: _int,
    ) -> "TTensor[_2DShape, T_DType, T_Device]": ...

    @overload
    def sum(  # type: ignore
        self,
        dim: Optional[_int] = None,
    ) -> "TTensor[_0DShape, T_DType, T_Device]": ...

    @overload
    def to(self, device: DeviceLike) -> "TTensor[T_Shape, T_DType, _AnyDevice]": ...  # type: ignore

    @overload
    def to(self, dtype: DTypeLike) -> "TTensor[T_Shape, _AnyDType, T_Device]": ...  # type: ignore

    @overload
    def to(self, *args, **kwargs) -> "TTensor[T_Shape, _AnyDType, _AnyDevice]": ...  # type: ignore

    @overload
    def tolist(self: "TTensor[_0DShape, dtypes.BoolDType, _AnyDevice]") -> _bool: ...  # type: ignore

    @overload
    def tolist(
        self: "TTensor[_0DShape, DTypeBase[L[False], L[False], Any], _AnyDevice]",
    ) -> _int: ...

    @overload
    def tolist(
        self: "TTensor[_0DShape, DTypeBase[L[False], L[True], L[True]], _AnyDevice]",
    ) -> _float: ...

    @overload
    def tolist(
        self: "TTensor[_0DShape, DTypeBase[L[True], L[False], L[False]], _AnyDevice]",
    ) -> _complex: ...

    @overload
    def tolist(
        self: "TTensor[_0DShape, _AnyDType, _AnyDevice]",
    ) -> BuiltinNumber: ...

    @overload
    def tolist(  # type: ignore
        self: "TTensor[_1DShape, dtypes.BoolDType, _AnyDevice]",
    ) -> List[_bool]: ...

    @overload
    def tolist(
        self: "TTensor[_1DShape, DTypeBase[L[False], L[False], Any], _AnyDevice]",
    ) -> List[_int]: ...

    @overload
    def tolist(
        self: "TTensor[_1DShape, DTypeBase[L[False], L[True], L[True]], _AnyDevice]",
    ) -> List[_float]: ...

    @overload
    def tolist(
        self: "TTensor[_1DShape, DTypeBase[L[True], L[False], L[False]], _AnyDevice]",
    ) -> List[_complex]: ...

    @overload
    def tolist(
        self: "TTensor[_1DShape, _AnyDType, _AnyDevice]",
    ) -> list: ...

    @overload
    def tolist(  # type: ignore
        self: "TTensor[_2DShape, dtypes.BoolDType, _AnyDevice]",
    ) -> List[List[_bool]]: ...

    @overload
    def tolist(
        self: "TTensor[_2DShape, DTypeBase[L[False], L[False], Any], _AnyDevice]",
    ) -> List[List[_int]]: ...

    @overload
    def tolist(
        self: "TTensor[_2DShape, DTypeBase[L[False], L[True], L[True]], _AnyDevice]",
    ) -> List[List[_float]]: ...

    @overload
    def tolist(
        self: "TTensor[_2DShape, DTypeBase[L[True], L[False], L[False]], _AnyDevice]",
    ) -> List[List[_complex]]: ...

    @overload
    def tolist(
        self: "TTensor[_2DShape, _AnyDType, _AnyDevice]",
    ) -> List[list]: ...

    @overload
    def tolist(  # type: ignore
        self: "TTensor[_3DShape, dtypes.BoolDType, _AnyDevice]",
    ) -> List[List[List[_bool]]]: ...

    @overload
    def tolist(
        self: "TTensor[_3DShape, DTypeBase[L[False], L[False], Any], _AnyDevice]",
    ) -> List[List[List[_int]]]: ...

    @overload
    def tolist(
        self: "TTensor[_3DShape, DTypeBase[L[False], L[True], L[True]], _AnyDevice]",
    ) -> List[List[List[_float]]]: ...

    @overload
    def tolist(
        self: "TTensor[_3DShape, DTypeBase[L[True], L[False], L[False]], _AnyDevice]",
    ) -> List[List[List[_complex]]]: ...

    @overload
    def tolist(
        self: "TTensor[_3DShape, _AnyDType, _AnyDevice]",
    ) -> List[List[list]]: ...

    @overload
    def tolist(self) -> Any: ...

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
    to = torch.Tensor.to  # noqa: F811  # type: ignore
    tolist = torch.Tensor.tolist  # noqa: F811  # type: ignore
    unsqueeze = torch.Tensor.unsqueeze  # noqa: F811  # type: ignore
    view = torch.Tensor.view  # noqa: F811  # type: ignore
