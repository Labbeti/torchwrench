#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Generic

from .base import (
    DeviceEnum,
    DTypeEnum,
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
x = Tensor2D[DTypeEnum.f32, DeviceEnum.cuda]([[2, 3, 4], [5, 6, 7]])
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
