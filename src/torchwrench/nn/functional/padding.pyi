from typing import Callable, Iterable, Literal, NamedTuple

from torch import Tensor
from torch.types import Number
from typing_extensions import TypeAlias

from torchwrench.core.make import DeviceLike as DeviceLike
from torchwrench.core.make import DTypeLike as DTypeLike
from torchwrench.core.make import GeneratorLike as GeneratorLike
from torchwrench.core.make import as_device as as_device
from torchwrench.core.make import as_dtype as as_dtype
from torchwrench.core.make import as_generator as as_generator
from torchwrench.types import Tensor0D as Tensor0D
from torchwrench.types import is_number_like as is_number_like

PadAlign: TypeAlias
PadValue: TypeAlias = Number | Tensor0D | Callable[[Tensor], Number]
PadMode: TypeAlias

class PaddedValue(NamedTuple):
    padded: Tensor
    lengths: Tensor

def pad_dim(
    x: Tensor,
    target_length: int,
    *,
    dim: int = -1,
    align: PadAlign = "left",
    pad_value: PadValue = 0.0,
    mode: PadMode = "constant",
    generator: GeneratorLike = None,
) -> Tensor: ...
def pad_dims(
    x: Tensor,
    target_lengths: Iterable[int],
    *,
    dims: Iterable[int] | Literal["auto"] | None = None,
    aligns: PadAlign | Iterable[PadAlign] = "left",
    pad_value: PadValue = 0.0,
    mode: PadMode = "constant",
    generator: GeneratorLike = None,
) -> Tensor: ...
def pad_and_stack_rec(
    sequence: Tensor | int | float | tuple | list,
    pad_value: Number = 0,
    *,
    align: PadAlign = "left",
    device: DeviceLike = None,
    dtype: DTypeLike = None,
) -> Tensor: ...
def collate_tensors(
    tensors: Iterable[Tensor], pad_value: float = 0.0, dim: int = -1
) -> PaddedValue: ...
def cat_padded_batch(
    x1: Tensor,
    x1_lens: Tensor,
    x2: Tensor,
    x2_lens: Tensor,
    seq_dim: int = -1,
    batch_dim: int = 0,
) -> tuple[Tensor, Tensor]: ...
