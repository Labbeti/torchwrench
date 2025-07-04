#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Literal, Optional, Sequence, Tuple, Union, overload

import torch
from pythonwrench.semver import Version
from pythonwrench.typing import BuiltinNumber
from torch import Tensor
from torch.types import Number
from typing_extensions import Never

from torchwrench.core.dtype_enum import DTypeEnum
from torchwrench.core.make import (
    DeviceLike,
    DTypeLike,
    GeneratorLike,
    as_device,
    as_dtype,
    as_generator,
)
from torchwrench.types import (
    BoolTensor0D,
    BoolTensor1D,
    BoolTensor2D,
    BoolTensor3D,
    CFloatTensor0D,
    CFloatTensor1D,
    CFloatTensor2D,
    CFloatTensor3D,
    FloatTensor0D,
    FloatTensor1D,
    FloatTensor2D,
    FloatTensor3D,
    LongTensor0D,
    LongTensor1D,
    LongTensor2D,
    LongTensor3D,
    Tensor0D,
    Tensor1D,
    Tensor2D,
    Tensor3D,
)

__all__ = [
    "arange",
    "empty",
    "full",
    "rand",
    "randint",
    "randperm",
    "ones",
    "zeros",
]

# ----------
# arange
# ----------


@overload
def arange(
    end: Number,
    *,
    out: Optional[Tensor] = None,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> LongTensor1D: ...


@overload
def arange(
    start: Number,
    end: Number,
    *,
    out: Optional[Tensor] = None,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> LongTensor1D: ...


@overload
def arange(
    start: Number,
    end: Number,
    step: Number,
    *,
    out: Optional[Tensor] = None,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> LongTensor1D: ...


@overload
def arange(
    end: Number,
    *,
    out: Optional[Tensor] = None,
    dtype: Union[torch.dtype, str, DTypeEnum],
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor1D: ...


@overload
def arange(
    start: Number,
    end: Number,
    *,
    out: Optional[Tensor] = None,
    dtype: Union[torch.dtype, str, DTypeEnum],
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor1D: ...


@overload
def arange(
    start: Number,
    end: Number,
    step: Number,
    *,
    out: Optional[Tensor] = None,
    dtype: Union[torch.dtype, str, DTypeEnum],
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor1D: ...


def arange(
    *args: Number,
    out: Optional[Tensor] = None,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
    **kwargs,
) -> torch.Tensor:
    dtype = as_dtype(dtype)
    device = as_device(device)

    kwds = {}
    if Version(torch.__version__) >= Version("2.0.0"):
        kwds.update(pin_memory=pin_memory)

    return torch.arange(
        *args,
        **kwargs,
        out=out,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        **kwds,
    )


# ----------
# empty
# ----------


@overload
def empty(
    size: Sequence[Never],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    memory_format: Optional[torch.memory_format] = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor0D: ...


@overload
def empty(
    size: Tuple[int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    memory_format: Optional[torch.memory_format] = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor1D: ...


@overload
def empty(
    size: Tuple[int, int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    memory_format: Optional[torch.memory_format] = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor2D: ...


@overload
def empty(
    size: Tuple[int, int, int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    memory_format: Optional[torch.memory_format] = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor3D: ...


@overload
def empty(
    size0: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    memory_format: Optional[torch.memory_format] = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor1D: ...


@overload
def empty(
    size0: int,
    size1: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    memory_format: Optional[torch.memory_format] = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor2D: ...


@overload
def empty(
    size0: int,
    size1: int,
    size2: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    memory_format: Optional[torch.memory_format] = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor3D: ...


def empty(
    *data,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    memory_format: Optional[torch.memory_format] = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> torch.Tensor:
    dtype = as_dtype(dtype)
    device = as_device(device)

    if layout is None:
        layout = torch.strided
    if pin_memory is None:
        pin_memory = False
    if requires_grad is None:
        requires_grad = False

    return torch.empty(  # type: ignore
        *data,
        memory_format=memory_format,
        out=out,
        dtype=dtype,  # type: ignore
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


# ----------
# full
# ----------


# bool
@overload
def full(  # type: ignore
    size: Sequence[Never],
    fill_value: bool,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> BoolTensor0D: ...


@overload
def full(  # type: ignore
    size: Tuple[int],
    fill_value: bool,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> BoolTensor1D: ...


@overload
def full(  # type: ignore
    size: Tuple[int, int],
    fill_value: bool,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> BoolTensor2D: ...


@overload
def full(  # type: ignore
    size: Tuple[int, int, int],
    fill_value: bool,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> BoolTensor3D: ...


# int
@overload
def full(
    size: Sequence[Never],
    fill_value: int,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> LongTensor0D: ...


@overload
def full(
    size: Tuple[int],
    fill_value: int,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> LongTensor1D: ...


@overload
def full(
    size: Tuple[int, int],
    fill_value: int,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> LongTensor2D: ...


@overload
def full(
    size: Tuple[int, int, int],
    fill_value: int,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> LongTensor3D: ...


# float
@overload
def full(
    size: Sequence[Never],
    fill_value: float,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> FloatTensor0D: ...


@overload
def full(
    size: Tuple[int],
    fill_value: float,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> FloatTensor1D: ...


@overload
def full(
    size: Tuple[int, int],
    fill_value: float,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> FloatTensor2D: ...


@overload
def full(
    size: Tuple[int, int, int],
    fill_value: float,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> FloatTensor3D: ...


# complex
@overload
def full(
    size: Sequence[Never],
    fill_value: complex,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> CFloatTensor0D: ...


@overload
def full(
    size: Tuple[int],
    fill_value: complex,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> CFloatTensor1D: ...


@overload
def full(
    size: Tuple[int, int],
    fill_value: complex,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> CFloatTensor2D: ...


@overload
def full(
    size: Tuple[int, int, int],
    fill_value: complex,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> CFloatTensor3D: ...


# BuiltinNumber
@overload
def full(
    size: Sequence[Never],
    fill_value: BuiltinNumber,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor0D: ...


@overload
def full(
    size: Tuple[int],
    fill_value: BuiltinNumber,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor1D: ...


@overload
def full(
    size: Tuple[int, int],
    fill_value: BuiltinNumber,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor2D: ...


@overload
def full(
    size: Tuple[int, int, int],
    fill_value: BuiltinNumber,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor3D: ...


def full(
    size: Sequence[int],
    fill_value: BuiltinNumber,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> torch.Tensor:
    dtype = as_dtype(dtype)
    device = as_device(device)
    kwds = {}
    if Version(torch.__version__) >= Version("2.0.0"):
        kwds.update(pin_memory=pin_memory)
    if layout is None:
        layout = torch.strided
    if requires_grad is None:
        requires_grad = False

    return torch.full(
        tuple(size),
        fill_value,  # type: ignore
        out=out,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
        **kwds,
    )


# ----------
# ones
# ----------


@overload
def ones(
    size: Sequence[Never],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor0D: ...


@overload
def ones(
    size: Tuple[int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor1D: ...


@overload
def ones(
    size: Tuple[int, int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor2D: ...


@overload
def ones(
    size: Tuple[int, int, int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor3D: ...


@overload
def ones(
    size0: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor1D: ...


@overload
def ones(
    size0: int,
    size1: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor2D: ...


@overload
def ones(
    size0: int,
    size1: int,
    size2: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor3D: ...


def ones(
    *data,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> torch.Tensor:
    dtype = as_dtype(dtype)
    device = as_device(device)

    if layout is None:
        layout = torch.strided
    if pin_memory is None:
        pin_memory = False
    if requires_grad is None:
        requires_grad = False

    return torch.ones(  # type: ignore
        *data,
        out=out,
        dtype=dtype,  # type: ignore
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


# ----------
# rand
# ----------
@overload
def rand(
    size: Sequence[Never],
    /,
    *,
    dtype: Literal["float", "float32", None] = None,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> FloatTensor0D: ...


@overload
def rand(
    size: Tuple[int],
    /,
    *,
    dtype: Literal["float", "float32", None] = None,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> FloatTensor1D: ...


@overload
def rand(
    size: Tuple[int, int],
    /,
    *,
    dtype: Literal["float", "float32", None] = None,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> FloatTensor2D: ...


@overload
def rand(
    size: Tuple[int, int, int],
    /,
    *,
    dtype: Literal["float", "float32", None] = None,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> FloatTensor3D: ...


@overload
def rand(
    size0: int,
    /,
    *,
    dtype: Literal["float", "float32", None] = None,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> FloatTensor1D: ...


@overload
def rand(
    size0: int,
    size1: int,
    /,
    *,
    dtype: Literal["float", "float32", None] = None,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> FloatTensor2D: ...


@overload
def rand(
    size0: int,
    size1: int,
    size2: int,
    /,
    *,
    dtype: Literal["float", "float32", None] = None,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> FloatTensor3D: ...


@overload
def rand(
    size: Sequence[Never],
    /,
    *,
    dtype: DTypeLike,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor0D: ...


@overload
def rand(
    size: Tuple[int],
    /,
    *,
    dtype: DTypeLike,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor1D: ...


@overload
def rand(
    size: Tuple[int, int],
    /,
    *,
    dtype: DTypeLike,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor2D: ...


@overload
def rand(
    size: Tuple[int, int, int],
    /,
    *,
    dtype: DTypeLike,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor3D: ...


@overload
def rand(
    size0: int,
    /,
    *,
    dtype: DTypeLike,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor1D: ...


@overload
def rand(
    size0: int,
    size1: int,
    /,
    *,
    dtype: DTypeLike,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor2D: ...


@overload
def rand(
    size0: int,
    size1: int,
    size2: int,
    /,
    *,
    dtype: DTypeLike,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor3D: ...


def rand(
    *data: Any,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> torch.Tensor:
    dtype = as_dtype(dtype)
    device = as_device(device)
    generator = as_generator(generator)

    kwds = {}
    if Version(torch.__version__) >= Version("2.0.0"):
        kwds.update(pin_memory=pin_memory)
    if layout is None:
        layout = torch.strided
    if requires_grad is None:
        requires_grad = False

    return torch.rand(  # type: ignore
        *data,
        generator=generator,
        out=out,
        dtype=dtype,  # type: ignore
        layout=layout,
        device=device,
        requires_grad=requires_grad,
        **kwds,
    )


# ----------
# randint
# ----------


@overload
def randint(
    low: int,
    high: int,
    size: Tuple[()],
    *,
    generator: GeneratorLike = None,
    dtype: Literal[None, "long", "int64"] = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> LongTensor0D: ...


@overload
def randint(
    low: int,
    high: int,
    size: Tuple[int],
    *,
    generator: GeneratorLike = None,
    dtype: Literal[None, "long", "int64"] = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> LongTensor1D: ...


@overload
def randint(
    low: int,
    high: int,
    size: Tuple[int, int],
    *,
    generator: GeneratorLike = None,
    dtype: Literal[None, "long", "int64"] = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> LongTensor2D: ...


@overload
def randint(
    low: int,
    high: int,
    size: Tuple[int, int, int],
    *,
    generator: GeneratorLike = None,
    dtype: Literal[None, "long", "int64"] = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> LongTensor3D: ...


@overload
def randint(
    low: int,
    high: int,
    size: Tuple[()],
    *,
    generator: GeneratorLike = None,
    dtype: Union[torch.dtype, str, DTypeLike],
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor0D: ...


@overload
def randint(
    low: int,
    high: int,
    size: Tuple[int],
    *,
    generator: GeneratorLike = None,
    dtype: Union[torch.dtype, str, DTypeLike],
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor1D: ...


@overload
def randint(
    low: int,
    high: int,
    size: Tuple[int, int],
    *,
    generator: GeneratorLike = None,
    dtype: Union[torch.dtype, str, DTypeLike],
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor2D: ...


@overload
def randint(
    low: int,
    high: int,
    size: Tuple[int, int, int],
    *,
    generator: GeneratorLike = None,
    dtype: Union[torch.dtype, str, DTypeLike],
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor3D: ...


def randint(
    low: int,
    high: int,
    size: Tuple[int, ...],
    *,
    generator: GeneratorLike = None,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor:
    dtype = as_dtype(dtype)
    device = as_device(device)
    generator = as_generator(generator)

    kwds = {}
    if Version(torch.__version__) >= Version("2.0.0"):
        kwds.update(pin_memory=pin_memory)

    return torch.randint(
        low=low,
        high=high,
        size=size,
        generator=generator,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        **kwds,
    )


# ----------
# randperm
# ----------


@overload
def randperm(
    n: int,
    *,
    generator: GeneratorLike = None,
    out: Optional[Tensor] = None,
    dtype: Literal[None, "long", "int64"] = "long",
    layout: Optional[torch.layout] = None,
    device: DeviceLike = None,
    pin_memory: Optional[bool] = False,
    requires_grad: Optional[bool] = False,
) -> LongTensor1D: ...


@overload
def randperm(
    n: int,
    *,
    generator: GeneratorLike = None,
    out: Optional[Tensor] = None,
    dtype: Union[torch.dtype, str, DTypeLike],
    layout: Optional[torch.layout] = None,
    device: DeviceLike = None,
    pin_memory: Optional[bool] = False,
    requires_grad: Optional[bool] = False,
) -> Tensor1D: ...


def randperm(
    n: int,
    *,
    generator: GeneratorLike = None,
    out: Optional[Tensor] = None,
    dtype: DTypeLike = "long",
    layout: Optional[torch.layout] = None,
    device: DeviceLike = None,
    pin_memory: Optional[bool] = False,
    requires_grad: Optional[bool] = False,
) -> Tensor:
    dtype = as_dtype(dtype)
    device = as_device(device)
    generator = as_generator(generator)

    if layout is None:
        layout = torch.strided
    if pin_memory is None:
        pin_memory = False
    if requires_grad is None:
        requires_grad = False

    return torch.randperm(
        n=n,
        generator=generator,
        out=out,
        dtype=dtype,  # type: ignore
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


# ----------
# zeros
# ----------
@overload
def zeros(
    size: Sequence[Never],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor0D: ...


@overload
def zeros(
    size: Tuple[int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor1D: ...


@overload
def zeros(
    size: Tuple[int, int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor2D: ...


@overload
def zeros(
    size: Tuple[int, int, int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor3D: ...


@overload
def zeros(
    size0: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor1D: ...


@overload
def zeros(
    size0: int,
    size1: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor2D: ...


@overload
def zeros(
    size0: int,
    size1: int,
    size2: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor3D: ...


def zeros(
    *data,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> torch.Tensor:
    dtype = as_dtype(dtype)
    device = as_device(device)

    if layout is None:
        layout = torch.strided
    if pin_memory is None:
        pin_memory = False
    if requires_grad is None:
        requires_grad = False

    return torch.zeros(  # type: ignore
        *data,
        out=out,
        dtype=dtype,  # type: ignore
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )
