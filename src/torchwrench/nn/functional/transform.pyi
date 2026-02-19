from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    overload,
)

import torch
from _typeshed import Incomplete
from pythonwrench.functools import function_alias
from pythonwrench.functools import identity as identity
from pythonwrench.typing import (
    BuiltinNumber,
    SupportsIterLen,
)
from pythonwrench.typing import (
    BuiltinScalar as BuiltinScalar,
)
from pythonwrench.typing import (
    T_BuiltinScalar as T_BuiltinScalar,
)
from torch import Tensor, nn
from typing_extensions import Never

from torchwrench.core.make import (
    DeviceLike as DeviceLike,
)
from torchwrench.core.make import (
    DTypeLike as DTypeLike,
)
from torchwrench.core.make import (
    GeneratorLike as GeneratorLike,
)
from torchwrench.core.make import (
    as_device as as_device,
)
from torchwrench.core.make import (
    as_dtype as as_dtype,
)
from torchwrench.core.make import (
    as_generator as as_generator,
)
from torchwrench.extras.numpy import (
    np as np,
)
from torchwrench.extras.numpy import (
    numpy_view_as_complex as numpy_view_as_complex,
)
from torchwrench.extras.numpy import (
    numpy_view_as_real as numpy_view_as_real,
)
from torchwrench.nn.functional.cropping import crop_dim as crop_dim
from torchwrench.nn.functional.others import nelement as nelement
from torchwrench.nn.functional.padding import (
    PadMode as PadMode,
)
from torchwrench.nn.functional.padding import (
    PadValue as PadValue,
)
from torchwrench.nn.functional.padding import (
    pad_dim as pad_dim,
)
from torchwrench.types import (
    ComplexFloatingTensor as ComplexFloatingTensor,
)
from torchwrench.types import (
    is_builtin_number as is_builtin_number,
)
from torchwrench.types import (
    is_scalar_like as is_scalar_like,
)
from torchwrench.types._typing import (
    BoolTensor0D as BoolTensor0D,
)
from torchwrench.types._typing import (
    BoolTensor1D as BoolTensor1D,
)
from torchwrench.types._typing import (
    BoolTensor2D as BoolTensor2D,
)
from torchwrench.types._typing import (
    BoolTensor3D as BoolTensor3D,
)
from torchwrench.types._typing import (
    CFloatTensor0D as CFloatTensor0D,
)
from torchwrench.types._typing import (
    CFloatTensor1D as CFloatTensor1D,
)
from torchwrench.types._typing import (
    CFloatTensor2D as CFloatTensor2D,
)
from torchwrench.types._typing import (
    CFloatTensor3D as CFloatTensor3D,
)
from torchwrench.types._typing import (
    FloatTensor0D as FloatTensor0D,
)
from torchwrench.types._typing import (
    FloatTensor1D as FloatTensor1D,
)
from torchwrench.types._typing import (
    FloatTensor2D as FloatTensor2D,
)
from torchwrench.types._typing import (
    FloatTensor3D as FloatTensor3D,
)
from torchwrench.types._typing import (
    LongTensor as LongTensor,
)
from torchwrench.types._typing import (
    LongTensor0D as LongTensor0D,
)
from torchwrench.types._typing import (
    LongTensor1D as LongTensor1D,
)
from torchwrench.types._typing import (
    LongTensor2D as LongTensor2D,
)
from torchwrench.types._typing import (
    LongTensor3D as LongTensor3D,
)
from torchwrench.types._typing import (
    ScalarLike as ScalarLike,
)
from torchwrench.types._typing import (
    T_Tensor as T_Tensor,
)
from torchwrench.types._typing import (
    T_TensorOrArray as T_TensorOrArray,
)
from torchwrench.types._typing import (
    Tensor0D as Tensor0D,
)
from torchwrench.types._typing import (
    Tensor1D as Tensor1D,
)
from torchwrench.types._typing import (
    Tensor2D as Tensor2D,
)
from torchwrench.types._typing import (
    Tensor3D as Tensor3D,
)
from torchwrench.utils import return_types as return_types

T = TypeVar("T")
U = TypeVar("U")
PadCropAlign: Incomplete
SqueezeMode: Incomplete

def repeat_interleave_nd(x: Tensor, repeats: int, dim: int = 0) -> Tensor: ...
def resample_nearest_rates(
    x: Tensor,
    rates: float | Iterable[float],
    *,
    dims: int | Iterable[int] = -1,
    round_fn: Callable[[Tensor], Tensor] = ...,
) -> Tensor: ...
def resample_nearest_freqs(
    x: Tensor,
    orig_freq: int,
    new_freq: int,
    *,
    dims: int | Iterable[int] = -1,
    round_fn: Callable[[Tensor], Tensor] = ...,
) -> Tensor: ...
def resample_nearest_steps(
    x: Tensor,
    steps: float | Iterable[float],
    *,
    dims: int | Iterable[int] = -1,
    round_fn: Callable[[Tensor], Tensor] = ...,
) -> Tensor: ...
def transform_drop(
    transform: Callable[[T], T], x: T, p: float, *, generator: GeneratorLike = None
) -> T: ...
def pad_and_crop_dim(
    x: Tensor,
    target_length: int,
    *,
    align: PadCropAlign = "left",
    pad_value: PadValue = 0.0,
    dim: int = -1,
    mode: PadMode = "constant",
    generator: GeneratorLike = None,
) -> Tensor: ...
def shuffled(
    x: T_Tensor, dims: int | Iterable[int] = -1, generator: GeneratorLike = None
) -> T_Tensor: ...
@overload
def flatten(x: Tensor, start_dim: int = 0, end_dim: int | None = None) -> Tensor1D: ...
@overload
def flatten(
    x: np.ndarray | np.generic, start_dim: int = 0, end_dim: int | None = None
) -> np.ndarray: ...
@overload
def flatten(
    x: T_BuiltinScalar, start_dim: int = 0, end_dim: int | None = None
) -> list[T_BuiltinScalar]: ...
@overload
def flatten(
    x: Iterable[T_BuiltinScalar], start_dim: int = 0, end_dim: int | None = None
) -> list[T_BuiltinScalar]: ...
def squeeze(
    x: T_TensorOrArray,
    dim: int | Iterable[int] | None = None,
    mode: SqueezeMode = "view_if_possible",
) -> T_TensorOrArray: ...
def squeeze_(x: Tensor, dim: int | Iterable[int] | None = None) -> Tensor: ...
def squeeze_copy(
    x: T_TensorOrArray, dim: int | Iterable[int] | None = None
) -> T_TensorOrArray: ...
def unsqueeze(
    x: T_TensorOrArray, dim: int | Iterable[int], mode: SqueezeMode = "view_if_possible"
) -> T_TensorOrArray: ...
def unsqueeze_(x: Tensor, dim: int | Iterable[int]) -> Tensor: ...
def unsqueeze_copy(x: T_TensorOrArray, dim: int | Iterable[int]) -> T_TensorOrArray: ...
@overload
def to_item(x: T_BuiltinScalar) -> T_BuiltinScalar: ...
@overload
def to_item(x: Tensor | np.ndarray | SupportsIterLen) -> BuiltinScalar: ...
@overload
def view_as_real(x: Tensor) -> Tensor: ...
@overload
def view_as_real(x: np.ndarray) -> np.ndarray: ...
@overload
def view_as_real(x: complex) -> tuple[float, float]: ...
@overload
def view_as_complex(x: Tensor) -> ComplexFloatingTensor: ...
@overload
def view_as_complex(x: np.ndarray) -> np.ndarray: ...
@overload
def view_as_complex(x: tuple[float, float]) -> complex: ...
@overload
def move_to(
    x: Mapping[T, U],
    predicate: Callable[[Tensor | nn.Module], bool] | None = None,
    **kwargs,
) -> dict[T, U]: ...
@overload
def move_to(
    x: T, predicate: Callable[[Tensor | nn.Module], bool] | None = None, **kwargs
) -> T: ...
def move_to_rec(*args, **kwargs) -> None: ...
@function_alias
def recursive_to(*args, **kwargs) -> None: ...
@overload
def as_tensor(
    data: Sequence[Never], dtype: Literal[None] = None, device: DeviceLike = None
) -> Tensor1D: ...
@overload
def as_tensor(
    data: Sequence[Sequence[Never]],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> Tensor2D: ...
@overload
def as_tensor(
    data: Sequence[Sequence[Sequence[Never]]],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> Tensor3D: ...
@overload
def as_tensor(
    data: bool, dtype: Literal[None] = None, device: DeviceLike = None
) -> BoolTensor0D: ...
@overload
def as_tensor(
    data: Sequence[bool], dtype: Literal[None] = None, device: DeviceLike = None
) -> BoolTensor1D: ...
@overload
def as_tensor(
    data: Sequence[Sequence[bool]],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> BoolTensor2D: ...
@overload
def as_tensor(
    data: Sequence[Sequence[Sequence[bool]]],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> BoolTensor3D: ...
@overload
def as_tensor(
    data: BuiltinNumber, dtype: Literal["bool"], device: DeviceLike = None
) -> BoolTensor0D: ...
@overload
def as_tensor(
    data: Sequence[BuiltinNumber], dtype: Literal["bool"], device: DeviceLike = None
) -> BoolTensor1D: ...
@overload
def as_tensor(
    data: Sequence[Sequence[BuiltinNumber]],
    dtype: Literal["bool"],
    device: DeviceLike = None,
) -> BoolTensor2D: ...
@overload
def as_tensor(
    data: Sequence[Sequence[Sequence[BuiltinNumber]]],
    dtype: Literal["bool"],
    device: DeviceLike = None,
) -> BoolTensor3D: ...
@overload
def as_tensor(
    data: int, dtype: Literal[None] = None, device: DeviceLike = None
) -> LongTensor0D: ...
@overload
def as_tensor(
    data: Sequence[int], dtype: Literal[None] = None, device: DeviceLike = None
) -> LongTensor1D: ...
@overload
def as_tensor(
    data: Sequence[Sequence[int]],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> LongTensor2D: ...
@overload
def as_tensor(
    data: Sequence[Sequence[Sequence[int]]],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> LongTensor3D: ...
@overload
def as_tensor(
    data: BuiltinNumber, dtype: Literal["int64", "long"], device: DeviceLike = None
) -> LongTensor0D: ...
@overload
def as_tensor(
    data: Sequence[BuiltinNumber],
    dtype: Literal["int64", "long"],
    device: DeviceLike = None,
) -> LongTensor1D: ...
@overload
def as_tensor(
    data: Sequence[Sequence[BuiltinNumber]],
    dtype: Literal["int64", "long"],
    device: DeviceLike = None,
) -> LongTensor2D: ...
@overload
def as_tensor(
    data: Sequence[Sequence[Sequence[BuiltinNumber]]],
    dtype: Literal["int64", "long"],
    device: DeviceLike = None,
) -> LongTensor3D: ...
@overload
def as_tensor(
    data: float, dtype: Literal[None] = None, device: DeviceLike = None
) -> FloatTensor0D: ...
@overload
def as_tensor(
    data: Sequence[float], dtype: Literal[None] = None, device: DeviceLike = None
) -> FloatTensor1D: ...
@overload
def as_tensor(
    data: Sequence[Sequence[float]],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> FloatTensor2D: ...
@overload
def as_tensor(
    data: Sequence[Sequence[Sequence[float]]],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> FloatTensor3D: ...
@overload
def as_tensor(
    data: BuiltinNumber, dtype: Literal["float32", "float"], device: DeviceLike = None
) -> FloatTensor0D: ...
@overload
def as_tensor(
    data: Sequence[BuiltinNumber],
    dtype: Literal["float32", "float"],
    device: DeviceLike = None,
) -> FloatTensor1D: ...
@overload
def as_tensor(
    data: Sequence[Sequence[BuiltinNumber]],
    dtype: Literal["float32", "float"],
    device: DeviceLike = None,
) -> FloatTensor2D: ...
@overload
def as_tensor(
    data: Sequence[Sequence[Sequence[BuiltinNumber]]],
    dtype: Literal["float32", "float"],
    device: DeviceLike = None,
) -> FloatTensor3D: ...
@overload
def as_tensor(
    data: complex, dtype: Literal[None] = None, device: DeviceLike = None
) -> CFloatTensor0D: ...
@overload
def as_tensor(
    data: Sequence[complex], dtype: Literal[None] = None, device: DeviceLike = None
) -> CFloatTensor1D: ...
@overload
def as_tensor(
    data: Sequence[Sequence[complex]],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> CFloatTensor2D: ...
@overload
def as_tensor(
    data: Sequence[Sequence[Sequence[complex]]],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> CFloatTensor3D: ...
@overload
def as_tensor(
    data: BuiltinNumber,
    dtype: Literal["complex64", "cfloat"],
    device: DeviceLike = None,
) -> CFloatTensor0D: ...
@overload
def as_tensor(
    data: Sequence[BuiltinNumber],
    dtype: Literal["complex64", "cfloat"],
    device: DeviceLike = None,
) -> CFloatTensor1D: ...
@overload
def as_tensor(
    data: Sequence[Sequence[BuiltinNumber]],
    dtype: Literal["complex64", "cfloat"],
    device: DeviceLike = None,
) -> CFloatTensor2D: ...
@overload
def as_tensor(
    data: Sequence[Sequence[Sequence[BuiltinNumber]]],
    dtype: Literal["complex64", "cfloat"],
    device: DeviceLike = None,
) -> CFloatTensor3D: ...
@overload
def as_tensor(
    data: BuiltinNumber, dtype: DTypeLike = None, device: DeviceLike = None
) -> Tensor0D: ...
@overload
def as_tensor(
    data: Sequence[BuiltinNumber], dtype: DTypeLike = None, device: DeviceLike = None
) -> Tensor1D: ...
@overload
def as_tensor(
    data: Sequence[Sequence[BuiltinNumber]],
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor2D: ...
@overload
def as_tensor(
    data: Sequence[Sequence[Sequence[BuiltinNumber]]],
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor3D: ...
@overload
def as_tensor(
    data: Any, dtype: DTypeLike = None, device: DeviceLike = None
) -> torch.Tensor: ...
@overload
def topk(
    x: Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *,
    return_values: Literal[True] = True,
    return_indices: Literal[True] = True,
) -> return_types.topk: ...
@overload
def topk(
    x: T_Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *,
    return_values: Literal[True] = True,
    return_indices: Literal[False],
) -> T_Tensor: ...
@overload
def topk(
    x: Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *,
    return_values: Literal[False],
    return_indices: Literal[True] = True,
) -> LongTensor: ...
@overload
def topk(
    x: T_Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *,
    return_values: bool = True,
    return_indices: bool = True,
) -> T_Tensor | LongTensor | return_types.topk: ...
@overload
def top_p(
    x: Tensor,
    p: float,
    dim: int = -1,
    largest: bool = True,
    *,
    return_values: Literal[True] = True,
    return_indices: Literal[True] = True,
) -> return_types.top_p: ...
@overload
def top_p(
    x: T_Tensor,
    p: float,
    dim: int = -1,
    largest: bool = True,
    *,
    return_values: Literal[True] = True,
    return_indices: Literal[False],
) -> T_Tensor: ...
@overload
def top_p(
    x: Tensor,
    p: float,
    dim: int = -1,
    largest: bool = True,
    *,
    return_values: Literal[False],
    return_indices: Literal[True] = True,
) -> LongTensor: ...
@overload
def top_p(
    x: T_Tensor,
    p: float,
    dim: int = -1,
    largest: bool = True,
    *,
    return_values: bool = True,
    return_indices: bool = True,
) -> T_Tensor | LongTensor | return_types.top_p: ...
def to_tensor(*args, **kwargs) -> None: ...
def top_k(*args, **kwargs) -> None: ...
