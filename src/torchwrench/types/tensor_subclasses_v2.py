#!/usr/bin/env python
# -*- coding: utf-8 -*-

from builtins import bool as _bool
from enum import auto
from typing import Any, Generic, Literal, Tuple, get_args, get_origin

import torch
from pythonwrench.enum import StrEnum
from torch._C import _TensorMeta
from typing_extensions import Self, Type, TypeVar

from torchwrench.core.dtype_enum import DTypeEnum

_0DShape = Tuple[()]
_1DShape = Tuple[int]
_2DShape = Tuple[int, int]
_3DShape = Tuple[int, int, int]
_4DShape = Tuple[int, int, int, int]


class DeviceGeneric(StrEnum):
    unknown = auto()
    cuda = auto()
    cpu = auto()


_DefaultShape = Tuple[Any, ...]
_DefaultDType = Any
_DefaultDevice = Literal[DeviceGeneric.unknown]


T_Shape = TypeVar(
    "T_Shape",
    bound=Tuple[int, ...],
    default=_DefaultShape,
    covariant=True,
)
T_DType = TypeVar(
    "T_DType",
    bound=str,
    default=_DefaultDType,
    covariant=True,
)
T_Device = TypeVar(
    "T_Device",
    bound=DeviceGeneric,
    default=_DefaultDevice,
    covariant=True,
)


class _TTensorMeta(_TensorMeta):
    def __instancecheck__(self, instance: Any) -> _bool:
        """Called method to check isinstance(instance, self)"""
        if not isinstance(instance, torch.Tensor):
            return False

        orig_bases: tuple = self.__orig_bases__  # type: ignore
        # breakpoint()
        raise NotImplementedError

    def __subclasscheck__(self, subclass: Any) -> _bool:
        """Called method to check issubclass(subclass, cls)"""
        orig_bases: tuple = self.__orig_bases__  # type: ignore
        # breakpoint()
        raise NotImplementedError


class TTensor(
    Generic[T_Shape, T_DType, T_Device],
    torch.Tensor,
    metaclass=_TTensorMeta,
):
    def __new__(cls, *args, **kwargs) -> "TTensor":
        return torch.as_tensor(*args, **kwargs)  # type: ignore


class Tensor2D(
    Generic[T_DType, T_Device],
    TTensor[_2DShape, T_DType, T_Device],
): ...


class FloatTensor(
    Generic[T_Shape, T_Device],
    TTensor[T_Shape, DTypeEnum.f32, T_Device],
): ...


class FloatTensor2D(
    Generic[T_Device],
    TTensor[_2DShape, DTypeEnum.f32, T_Device],
): ...


class CudaFloatTensor2D(
    TTensor[_2DShape, DTypeEnum.f32, DeviceGeneric.cuda],
): ...
