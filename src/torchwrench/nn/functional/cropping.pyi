from typing import Iterable, Literal

from _typeshed import Incomplete
from torch import Tensor as Tensor

from torchwrench.core.make import (
    GeneratorLike as GeneratorLike,
)
from torchwrench.core.make import (
    as_generator as as_generator,
)

CropAlign: Incomplete

def crop_dim(
    x: Tensor,
    target_length: int,
    *,
    dim: int = -1,
    align: CropAlign = "left",
    generator: GeneratorLike = None,
) -> Tensor: ...
def crop_dims(
    x: Tensor,
    target_lengths: Iterable[int],
    *,
    dims: Iterable[int] | Literal["auto"] | None = "auto",
    aligns: CropAlign | Iterable[CropAlign] = "left",
    generator: GeneratorLike = None,
) -> Tensor: ...
