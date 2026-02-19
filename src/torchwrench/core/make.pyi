from typing import Literal, overload

import torch
from _typeshed import Incomplete
from torch.types import Device as Device
from typing_extensions import TypeAlias

from torchwrench.core.dtype_enum import (
    DTypeEnum as DTypeEnum,
)
from torchwrench.core.dtype_enum import (
    enum_dtype_to_torch_dtype as enum_dtype_to_torch_dtype,
)
from torchwrench.core.dtype_enum import (
    str_to_torch_dtype as str_to_torch_dtype,
)

DeviceLike: TypeAlias
DTypeLike: TypeAlias
GeneratorLike: TypeAlias
Generator: Incomplete
device: Incomplete
CUDA_IF_AVAILABLE: str

def get_default_device() -> torch.device: ...
def get_default_dtype() -> torch.dtype: ...
def get_default_generator() -> torch.Generator: ...
def set_default_dtype(dtype: DTypeLike) -> None: ...
def set_default_generator(generator: GeneratorLike) -> None: ...
@overload
def as_device(device: Literal[None]) -> None: ...
@overload
def as_device(device: str | int | torch.device = ...) -> torch.device: ...
@overload
def as_dtype(dtype: Literal[None] = None) -> None: ...
@overload
def as_dtype(dtype: str | DTypeEnum | torch.dtype) -> torch.dtype: ...
@overload
def as_generator(generator: Literal[None] = None) -> None: ...
@overload
def as_generator(
    generator: int | torch.Generator | Literal["default"],
) -> torch.Generator: ...
