#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Generic
from typing import Literal as L

from torchwrench.core import dtype_enum_v2 as dtypes
from torchwrench.types.typed_tensor.base import (
    DeviceEnum,
    T_Device,
    T_DType,
    T_Shape,
    TTensor,
    _0DShape,
    _1DShape,
    _2DShape,
    _3DShape,
)


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
    TTensor[T_Shape, dtypes.BoolDType, T_Device],
): ...


class BoolTensor0D(
    Generic[T_Device],
    TTensor[_0DShape, dtypes.BoolDType, T_Device],
): ...


class CPUTensor(
    Generic[T_Shape, T_DType],
    TTensor[T_Shape, T_DType, DeviceEnum.cpu],
): ...


class CUDAFloatTensor2D(
    TTensor[_2DShape, dtypes.FloatDType, DeviceEnum.cuda],
): ...


class DoubleTensor(
    Generic[T_Shape, T_Device],
    TTensor[T_Shape, dtypes.DoubleDType, T_Device],
): ...


class FloatTensor(
    Generic[T_Shape, T_Device],
    TTensor[T_Shape, dtypes.FloatDType, T_Device],
): ...


class FloatTensor2D(
    Generic[T_Device],
    TTensor[_2DShape, dtypes.FloatDType, T_Device],
): ...


class HalfTensor(
    Generic[T_Shape, T_Device],
    TTensor[T_Shape, dtypes.HalfDType, T_Device],
): ...


class IntTensor(
    Generic[T_Shape, T_Device],
    TTensor[T_Shape, dtypes.IntDType, T_Device],
): ...


class LongTensor(
    Generic[T_Shape, T_Device],
    TTensor[T_Shape, dtypes.LongDType, T_Device],
): ...


class ShortTensor(
    Generic[T_Shape, T_Device],
    TTensor[T_Shape, dtypes.ShortDType, T_Device],
): ...


class SignedIntegerTensor(
    Generic[T_Shape, T_Device],
    TTensor[T_Shape, dtypes.DTypeBase[L[False], L[False], L[True]], T_Device],
): ...


# TODO: rm
x = Tensor2D[dtypes.FloatDType, DeviceEnum.cpu]([[2, 3, 4], [5, 6, 7]])
m = x.ndim
z = x[0]
p = z.shape
a = z[0]
q = a.shape

y = x.view((3, 2, 1))
s = y.shape
n = y.ndim
o = y[0]
r = o[None]

b = x[None] == y

c = x.double()
d = x.short()

e = x.isfinite()
f = x.isinf()

g = x.mean()
h = x.sum().isinf()

i = g.ndim
j = h.ndim

k = x == c

is_signed = x.is_signed()
m = TTensor[_0DShape, dtypes.UInt32DType, DeviceEnum.cpu]()
is_signed = m.is_signed()
is_complex = m.is_complex()
is_floating_point = m.is_floating_point()

n = m.bool().item()

o = TTensor[Any, dtypes.BoolDType](1)
p = o.item()
q = o.int()

print(isinstance(o, SignedIntegerTensor))
print(isinstance(q, SignedIntegerTensor))
