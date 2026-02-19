from typing import Literal, Sequence, overload

import torch
from pythonwrench.typing import BuiltinNumber
from torch import Tensor
from torch.types import Number
from typing_extensions import Never

from torchwrench.core.dtype_enum import DTypeEnum
from torchwrench.core.make import DeviceLike, DTypeLike, GeneratorLike
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

__all__ = ["arange", "empty", "full", "rand", "randint", "randperm", "ones", "zeros"]

@overload
def arange(
    end: Number,
    *,
    out: Tensor | None = None,
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
    out: Tensor | None = None,
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
    out: Tensor | None = None,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> LongTensor1D: ...
@overload
def arange(
    end: Number,
    *,
    out: Tensor | None = None,
    dtype: torch.dtype | str | DTypeEnum,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor1D: ...
@overload
def arange(
    start: Number,
    end: Number,
    *,
    out: Tensor | None = None,
    dtype: torch.dtype | str | DTypeEnum,
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
    out: Tensor | None = None,
    dtype: torch.dtype | str | DTypeEnum,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor1D: ...
@overload
def empty(
    size: Sequence[Never],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    memory_format: torch.memory_format | None = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor0D: ...
@overload
def empty(
    size: tuple[int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    memory_format: torch.memory_format | None = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor1D: ...
@overload
def empty(
    size: tuple[int, int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    memory_format: torch.memory_format | None = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor2D: ...
@overload
def empty(
    size: tuple[int, int, int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    memory_format: torch.memory_format | None = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor3D: ...
@overload
def empty(
    size0: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    memory_format: torch.memory_format | None = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor1D: ...
@overload
def empty(
    size0: int,
    size1: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    memory_format: torch.memory_format | None = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
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
    memory_format: torch.memory_format | None = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor3D: ...
@overload
def full(
    size: Sequence[Never],
    fill_value: bool,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> BoolTensor0D: ...
@overload
def full(
    size: tuple[int],
    fill_value: bool,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> BoolTensor1D: ...
@overload
def full(
    size: tuple[int, int],
    fill_value: bool,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> BoolTensor2D: ...
@overload
def full(
    size: tuple[int, int, int],
    fill_value: bool,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> BoolTensor3D: ...
@overload
def full(
    size: Sequence[Never],
    fill_value: int,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> LongTensor0D: ...
@overload
def full(
    size: tuple[int],
    fill_value: int,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> LongTensor1D: ...
@overload
def full(
    size: tuple[int, int],
    fill_value: int,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> LongTensor2D: ...
@overload
def full(
    size: tuple[int, int, int],
    fill_value: int,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> LongTensor3D: ...
@overload
def full(
    size: Sequence[Never],
    fill_value: float,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> FloatTensor0D: ...
@overload
def full(
    size: tuple[int],
    fill_value: float,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> FloatTensor1D: ...
@overload
def full(
    size: tuple[int, int],
    fill_value: float,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> FloatTensor2D: ...
@overload
def full(
    size: tuple[int, int, int],
    fill_value: float,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> FloatTensor3D: ...
@overload
def full(
    size: Sequence[Never],
    fill_value: complex,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> CFloatTensor0D: ...
@overload
def full(
    size: tuple[int],
    fill_value: complex,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> CFloatTensor1D: ...
@overload
def full(
    size: tuple[int, int],
    fill_value: complex,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> CFloatTensor2D: ...
@overload
def full(
    size: tuple[int, int, int],
    fill_value: complex,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> CFloatTensor3D: ...
@overload
def full(
    size: Sequence[Never],
    fill_value: BuiltinNumber,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor0D: ...
@overload
def full(
    size: tuple[int],
    fill_value: BuiltinNumber,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor1D: ...
@overload
def full(
    size: tuple[int, int],
    fill_value: BuiltinNumber,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor2D: ...
@overload
def full(
    size: tuple[int, int, int],
    fill_value: BuiltinNumber,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor3D: ...
@overload
def ones(
    size: Sequence[Never],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor0D: ...
@overload
def ones(
    size: tuple[int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor1D: ...
@overload
def ones(
    size: tuple[int, int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor2D: ...
@overload
def ones(
    size: tuple[int, int, int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor3D: ...
@overload
def ones(
    size0: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor1D: ...
@overload
def ones(
    size0: int,
    size1: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
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
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor3D: ...
@overload
def rand(
    size: Sequence[Never],
    /,
    *,
    dtype: Literal["float", "float32", None] = None,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> FloatTensor0D: ...
@overload
def rand(
    size: tuple[int],
    /,
    *,
    dtype: Literal["float", "float32", None] = None,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> FloatTensor1D: ...
@overload
def rand(
    size: tuple[int, int],
    /,
    *,
    dtype: Literal["float", "float32", None] = None,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> FloatTensor2D: ...
@overload
def rand(
    size: tuple[int, int, int],
    /,
    *,
    dtype: Literal["float", "float32", None] = None,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> FloatTensor3D: ...
@overload
def rand(
    size0: int,
    /,
    *,
    dtype: Literal["float", "float32", None] = None,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
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
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
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
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> FloatTensor3D: ...
@overload
def rand(
    size: Sequence[Never],
    /,
    *,
    dtype: DTypeLike,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor0D: ...
@overload
def rand(
    size: tuple[int],
    /,
    *,
    dtype: DTypeLike,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor1D: ...
@overload
def rand(
    size: tuple[int, int],
    /,
    *,
    dtype: DTypeLike,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor2D: ...
@overload
def rand(
    size: tuple[int, int, int],
    /,
    *,
    dtype: DTypeLike,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor3D: ...
@overload
def rand(
    size0: int,
    /,
    *,
    dtype: DTypeLike,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
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
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
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
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor3D: ...
@overload
def randint(
    low: int,
    high: int,
    size: tuple[()],
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
    size: tuple[int],
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
    size: tuple[int, int],
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
    size: tuple[int, int, int],
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
    size: tuple[()],
    *,
    generator: GeneratorLike = None,
    dtype: torch.dtype | str | DTypeLike,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor0D: ...
@overload
def randint(
    low: int,
    high: int,
    size: tuple[int],
    *,
    generator: GeneratorLike = None,
    dtype: torch.dtype | str | DTypeLike,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor1D: ...
@overload
def randint(
    low: int,
    high: int,
    size: tuple[int, int],
    *,
    generator: GeneratorLike = None,
    dtype: torch.dtype | str | DTypeLike,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor2D: ...
@overload
def randint(
    low: int,
    high: int,
    size: tuple[int, int, int],
    *,
    generator: GeneratorLike = None,
    dtype: torch.dtype | str | DTypeLike,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor3D: ...
@overload
def randperm(
    n: int,
    *,
    generator: GeneratorLike = None,
    out: Tensor | None = None,
    dtype: Literal[None, "long", "int64"] = "long",
    layout: torch.layout | None = None,
    device: DeviceLike = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> LongTensor1D: ...
@overload
def randperm(
    n: int,
    *,
    generator: GeneratorLike = None,
    out: Tensor | None = None,
    dtype: torch.dtype | str | DTypeLike,
    layout: torch.layout | None = None,
    device: DeviceLike = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor1D: ...
@overload
def zeros(
    size: Sequence[Never],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor0D: ...
@overload
def zeros(
    size: tuple[int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor1D: ...
@overload
def zeros(
    size: tuple[int, int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor2D: ...
@overload
def zeros(
    size: tuple[int, int, int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor3D: ...
@overload
def zeros(
    size0: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor1D: ...
@overload
def zeros(
    size0: int,
    size1: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
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
    out: torch.Tensor | None = None,
    layout: torch.layout | None = None,
    pin_memory: bool | None = False,
    requires_grad: bool | None = False,
) -> Tensor3D: ...
