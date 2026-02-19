from typing import Any, Callable, Generic, Iterable, TypeVar, overload

from _typeshed import Incomplete
from pythonwrench.typing import BuiltinScalar as BuiltinScalar
from pythonwrench.typing import SupportsIterLen as SupportsIterLen
from torch import Tensor as Tensor
from torch import nn as nn

from torchwrench.core.make import DeviceLike as DeviceLike
from torchwrench.core.make import DTypeLike as DTypeLike
from torchwrench.core.make import GeneratorLike as GeneratorLike
from torchwrench.extras.numpy import np as np
from torchwrench.nn.functional.transform import PadCropAlign as PadCropAlign
from torchwrench.nn.functional.transform import PadMode as PadMode
from torchwrench.nn.functional.transform import PadValue as PadValue
from torchwrench.nn.functional.transform import SqueezeMode as SqueezeMode
from torchwrench.nn.functional.transform import T_BuiltinScalar as T_BuiltinScalar
from torchwrench.nn.functional.transform import as_tensor as as_tensor
from torchwrench.nn.functional.transform import flatten as flatten
from torchwrench.nn.functional.transform import identity as identity
from torchwrench.nn.functional.transform import move_to_rec as move_to_rec
from torchwrench.nn.functional.transform import pad_and_crop_dim as pad_and_crop_dim
from torchwrench.nn.functional.transform import (
    repeat_interleave_nd as repeat_interleave_nd,
)
from torchwrench.nn.functional.transform import (
    resample_nearest_freqs as resample_nearest_freqs,
)
from torchwrench.nn.functional.transform import (
    resample_nearest_rates as resample_nearest_rates,
)
from torchwrench.nn.functional.transform import (
    resample_nearest_steps as resample_nearest_steps,
)
from torchwrench.nn.functional.transform import shuffled as shuffled
from torchwrench.nn.functional.transform import squeeze as squeeze
from torchwrench.nn.functional.transform import to_item as to_item
from torchwrench.nn.functional.transform import top_p as top_p
from torchwrench.nn.functional.transform import topk as topk
from torchwrench.nn.functional.transform import transform_drop as transform_drop
from torchwrench.nn.functional.transform import unsqueeze as unsqueeze
from torchwrench.nn.functional.transform import view_as_complex as view_as_complex
from torchwrench.nn.functional.transform import view_as_real as view_as_real
from torchwrench.types._typing import ComplexFloatingTensor as ComplexFloatingTensor
from torchwrench.types._typing import LongTensor as LongTensor
from torchwrench.types._typing import ScalarLike as ScalarLike
from torchwrench.types._typing import T_TensorOrArray as T_TensorOrArray
from torchwrench.utils import return_types as return_types

from .module import EModule as EModule
from .module import Module as Module

T = TypeVar("T")

class AsTensor(Module):
    device: Incomplete
    dtype: Incomplete
    def __init__(
        self, *, device: DeviceLike = None, dtype: DTypeLike = None
    ) -> None: ...
    def forward(self, x: Any) -> Tensor: ...
    def extra_repr(self) -> str: ...

class Flatten(Module):
    start_dim: Incomplete
    end_dim: Incomplete
    def __init__(self, start_dim: int = 0, end_dim: int | None = None) -> None: ...
    @overload
    def forward(self, x: Tensor) -> Tensor: ...
    @overload
    def forward(self, x: np.ndarray | np.generic) -> np.ndarray: ...
    @overload
    def forward(self, x: T_BuiltinScalar) -> list[T_BuiltinScalar]: ...
    @overload
    def forward(self, x: Iterable[T_BuiltinScalar]) -> list[T_BuiltinScalar]: ...
    def extra_repr(self) -> str: ...

class Identity(Module):
    def __init__(self, *args, **kwargs) -> None: ...
    def forward(self, x: T) -> T: ...

class MoveToRec(Module):
    predicate: Incomplete
    def __init__(
        self, predicate: Callable[[Tensor | nn.Module], bool] | None = None
    ) -> None: ...
    def forward(self, x: Any) -> Any: ...

class PadAndCropDim(Module):
    target_length: Incomplete
    align: PadCropAlign
    pad_value: Incomplete
    dim: Incomplete
    mode: PadMode
    generator: GeneratorLike
    def __init__(
        self,
        target_length: int,
        align: PadCropAlign = "left",
        pad_value: PadValue = 0.0,
        dim: int = -1,
        mode: PadMode = "constant",
        generator: GeneratorLike = None,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class RepeatInterleaveNd(Module):
    repeats: Incomplete
    dim: Incomplete
    def __init__(self, repeats: int, dim: int) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class ResampleNearestRates(Module):
    rates: Incomplete
    dims: Incomplete
    round_fn: Incomplete
    def __init__(
        self,
        rates: float | Iterable[float],
        dims: int | Iterable[int] = -1,
        round_fn: Callable[[Tensor], Tensor] = ...,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class ResampleNearestFreqs(Module):
    orig_freq: Incomplete
    new_freq: Incomplete
    dims: Incomplete
    round_fn: Incomplete
    def __init__(
        self,
        orig_freq: int,
        new_freq: int,
        dims: int | Iterable[int] = -1,
        round_fn: Callable[[Tensor], Tensor] = ...,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class ResampleNearestSteps(Module):
    steps: Incomplete
    dims: Incomplete
    round_fn: Incomplete
    def __init__(
        self,
        steps: float | Iterable[float],
        dims: int | Iterable[int] = -1,
        round_fn: Callable[[Tensor], Tensor] = ...,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class Squeeze(Module):
    dim: Incomplete
    mode: SqueezeMode
    def __init__(
        self,
        dim: int | Iterable[int] | None = None,
        mode: SqueezeMode = "view_if_possible",
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class Shuffled(Module):
    dims: Incomplete
    generator: GeneratorLike
    def __init__(
        self, dims: int | Iterable[int] = -1, generator: GeneratorLike = None
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class ToItem(Module):
    def forward(
        self, x: ScalarLike | Tensor | np.ndarray | SupportsIterLen
    ) -> BuiltinScalar: ...

class TransformDrop(EModule[T, T], Generic[T]):
    transform: Incomplete
    p: Incomplete
    generator: GeneratorLike
    def __init__(
        self, transform: Callable[[T], T], p: float, generator: GeneratorLike = None
    ) -> None: ...
    def forward(self, x: T) -> T: ...
    def extra_repr(self) -> str: ...

class Unsqueeze(Module):
    dim: Incomplete
    mode: SqueezeMode
    def __init__(
        self, dim: int | Iterable[int], mode: SqueezeMode = "view_if_possible"
    ) -> None: ...
    def forward(self, x: T_TensorOrArray) -> T_TensorOrArray: ...
    def extra_repr(self) -> str: ...

class ViewAsReal(Module):
    def forward(
        self, x: Tensor | np.ndarray | complex
    ) -> Tensor | np.ndarray | tuple[float, float]: ...

class ViewAsComplex(Module):
    def forward(
        self, x: Tensor | np.ndarray | tuple[float, float]
    ) -> ComplexFloatingTensor | np.ndarray | complex: ...

class Topk(Module):
    k: Incomplete
    dim: Incomplete
    largest: Incomplete
    sorted: Incomplete
    return_values: Incomplete
    return_indices: Incomplete
    def __init__(
        self,
        k: int,
        dim: int = -1,
        largest: bool = True,
        sorted: bool = True,
        *,
        return_values: bool = True,
        return_indices: bool = True,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor | LongTensor | return_types.topk: ...
    def extra_repr(self) -> str: ...

class TopP(Module):
    p: Incomplete
    dim: Incomplete
    largest: Incomplete
    return_values: Incomplete
    return_indices: Incomplete
    def __init__(
        self,
        p: float,
        dim: int = -1,
        largest: bool = True,
        *,
        return_values: bool = True,
        return_indices: bool = True,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor | LongTensor | return_types.top_p: ...
    def extra_repr(self) -> str: ...

ToTensor = AsTensor
TopK = Topk
