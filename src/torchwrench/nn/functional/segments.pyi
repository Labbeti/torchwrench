from torch import Tensor

from torchwrench.core.make import DeviceLike as DeviceLike
from torchwrench.core.make import as_device as as_device
from torchwrench.nn.functional.padding import (
    pad_and_stack_rec as pad_and_stack_rec,
)
from torchwrench.nn.functional.padding import (
    pad_dim as pad_dim,
)
from torchwrench.types import BoolTensor as BoolTensor
from torchwrench.types import LongTensor as LongTensor

def activity_to_segments(x: Tensor) -> LongTensor: ...
def segments_to_segments_list(
    segments: Tensor, maxsize: int | tuple[int, ...] | None = None
) -> list[tuple[int, int]] | list: ...
def segments_list_to_activity(
    segments_list: list[tuple[int, int]] | Tensor | list,
    maxsize: int | None = None,
    *,
    device: DeviceLike = None,
) -> BoolTensor: ...
def activity_to_segments_list(x: Tensor) -> list[tuple[int, int]] | list: ...
def segments_to_activity(x: Tensor, maxsize: int | None = None) -> BoolTensor: ...
