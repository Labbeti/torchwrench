#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod
from functools import lru_cache
from typing import Generic, List, Literal, Tuple, Type

import torch
from typing_extensions import TypeAlias, TypeVar

T_Complex = TypeVar("T_Complex", bound=bool, default=bool)
T_Floating = TypeVar("T_Floating", bound=bool, default=bool)
T_Signed = TypeVar("T_Signed", bound=bool, default=bool)


class DTypeBase(Generic[T_Complex, T_Floating, T_Signed]):
    @classmethod
    @abstractmethod
    def is_complex_dtype(cls) -> T_Complex:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def is_floating(cls) -> T_Floating:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def is_signed(cls) -> T_Signed:
        raise NotImplementedError


class ComplexDType(DTypeBase[Literal[True], Literal[False], Literal[True]]):
    @classmethod
    def is_complex_dtype(cls) -> Literal[True]:
        return True

    @classmethod
    def is_floating(cls) -> Literal[False]:
        return False

    @classmethod
    def is_signed(cls) -> Literal[True]:
        return True


class RealDType(
    Generic[T_Floating, T_Signed], DTypeBase[Literal[False], T_Floating, T_Signed]
):
    @classmethod
    def is_complex_dtype(cls) -> Literal[False]:
        return False


class FloatingDType(RealDType[Literal[True], Literal[True]]):
    @classmethod
    def is_floating(cls) -> Literal[True]:
        return True

    @classmethod
    def is_signed(cls) -> Literal[True]:
        return True


class IntegralDType(Generic[T_Signed], RealDType[Literal[False], T_Signed]):
    @classmethod
    def is_floating(cls) -> Literal[False]:
        return False


class SignedIntegerDType(IntegralDType[Literal[True]]):
    @classmethod
    def is_signed(cls) -> Literal[True]:
        return True


class UnsignedIntegerDType(IntegralDType[Literal[False]]):
    @classmethod
    def is_signed(cls) -> Literal[False]:
        return False


# Leaf types
class Float16DType(FloatingDType): ...


class Float32DType(FloatingDType): ...


class Float64DType(FloatingDType): ...


class Int16DType(SignedIntegerDType): ...


class Int32DType(SignedIntegerDType): ...


class Int64DType(SignedIntegerDType): ...


class Complex64DType(ComplexDType): ...


class Complex128DType(ComplexDType): ...


# Aliases types
HalfDType: TypeAlias = Float16DType
FloatDType: TypeAlias = Float32DType
DoubleDType: TypeAlias = Float64DType
ShortDType: TypeAlias = Int16DType
IntDType: TypeAlias = Int32DType
LongDType: TypeAlias = Int64DType
CFloatDType: TypeAlias = Complex64DType
CDoubleDType: TypeAlias = Complex128DType


class BFloat16DType(FloatingDType): ...


class BoolDType(UnsignedIntegerDType): ...


class Int8DType(SignedIntegerDType): ...


class QInt8DType(SignedIntegerDType): ...


class QInt32DType(SignedIntegerDType): ...


class QUInt4x2DType(UnsignedIntegerDType): ...


class QUInt8DType(UnsignedIntegerDType): ...


class UInt8DType(UnsignedIntegerDType): ...


# Only defined in some PyTorch versions
class Complex32DType(ComplexDType): ...


CHalfDType: TypeAlias = Complex32DType


class QUInt2x4DType(UnsignedIntegerDType): ...


class UInt16DType(UnsignedIntegerDType): ...


class UInt32DType(UnsignedIntegerDType): ...


class UInt64DType(UnsignedIntegerDType): ...


_DTYPES_TABLE: List[Tuple[str, torch.dtype, type]] = [
    # Leafs
    ("float16", torch.float16, Float16DType),
    ("float32", torch.float32, Float32DType),
    ("float64", torch.float64, Float64DType),
    ("int16", torch.int16, Int16DType),
    ("int32", torch.int32, Int32DType),
    ("int64", torch.int64, Int64DType),
    ("complex64", torch.complex64, Complex64DType),
    ("complex128", torch.complex128, Complex128DType),
    # Aliases
    ("half", torch.half, HalfDType),
    ("float", torch.float, FloatDType),
    ("double", torch.double, DoubleDType),
    ("short", torch.short, ShortDType),
    ("int", torch.int, IntDType),
    ("long", torch.long, LongDType),
    ("cfloat", torch.cfloat, CFloatDType),
    ("cdouble", torch.cdouble, CDoubleDType),
    # Others
    ("bfloat16", torch.bfloat16, BFloat16DType),
    ("bool", torch.bool, BoolDType),
    ("int8", torch.int8, Int8DType),
    ("qint8", torch.qint8, QInt8DType),
    ("qint32", torch.qint32, QInt32DType),
    ("quint4x2", torch.quint4x2, QUInt4x2DType),
    ("quint8", torch.quint8, QUInt8DType),
    ("uint8", torch.uint8, UInt8DType),
]


# Optional
if hasattr(torch, "complex32"):
    _DTYPES_TABLE += [("complex32", torch.complex32, Complex32DType)]

if hasattr(torch, "chalf"):
    _DTYPES_TABLE += [("chalf", torch.chalf, CHalfDType)]
elif hasattr(torch, "complex32"):
    _DTYPES_TABLE += [("chalf", torch.complex32, Complex32DType)]

if hasattr(torch, "quint2x4"):
    _DTYPES_TABLE += [("quint2x4", torch.quint2x4, QUInt2x4DType)]

if hasattr(torch, "uint16"):
    _DTYPES_TABLE += [("uint16", torch.uint16, UInt16DType)]

if hasattr(torch, "uint32"):
    _DTYPES_TABLE += [("uint32", torch.uint32, UInt32DType)]

if hasattr(torch, "uint64"):
    _DTYPES_TABLE += [("uint64", torch.uint64, UInt64DType)]


@lru_cache(len(_DTYPES_TABLE))
def dtype_cls_to_dtype(dtype_cls: Type[DTypeBase]) -> torch.dtype:
    for _, dtype_2, dtype_cls_2 in _DTYPES_TABLE:
        if dtype_cls == dtype_cls_2:
            return dtype_2

    msg = f"Invalid argument {dtype_cls=}."
    raise ValueError(msg)


@lru_cache(len(_DTYPES_TABLE))
def dtype_cls_to_dtype_name(dtype_cls: Type[DTypeBase]) -> str:
    for dtype_name_2, _, dtype_cls_2 in _DTYPES_TABLE:
        if dtype_cls == dtype_cls_2:
            return dtype_name_2

    msg = f"Invalid argument {dtype_cls=}."
    raise ValueError(msg)


@lru_cache(len(_DTYPES_TABLE))
def dtype_name_to_dtype(dtype_name: Type[DTypeBase]) -> torch.dtype:
    for dtype_name_2, dtype_2, _ in _DTYPES_TABLE:
        if dtype_name == dtype_name_2:
            return dtype_2

    msg = f"Invalid argument {dtype_name=}."
    raise ValueError(msg)


@lru_cache(len(_DTYPES_TABLE))
def dtype_name_to_dtype_cls(dtype_name: str) -> Type[DTypeBase]:
    for dtype_name_2, _, dtype_cls_2 in _DTYPES_TABLE:
        if dtype_name == dtype_name_2:
            return dtype_cls_2

    msg = f"Invalid argument {dtype_name=}."
    raise ValueError(msg)


@lru_cache(len(_DTYPES_TABLE))
def dtype_to_dtype_cls(dtype: torch.dtype) -> Type[DTypeBase]:
    for _, dtype_2, dtype_cls_2 in _DTYPES_TABLE:
        if dtype == dtype_2:
            return dtype_cls_2

    msg = f"Invalid argument {dtype=}."
    raise ValueError(msg)
