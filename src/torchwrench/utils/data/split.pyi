from typing import Callable, Iterable

from torch import Tensor as Tensor

from torchwrench.core.make import GeneratorLike as GeneratorLike
from torchwrench.core.make import as_generator as as_generator
from torchwrench.nn.functional.transform import as_tensor as as_tensor

def random_split(
    num_samples_or_indices: int | list[int] | Tensor,
    lengths: Iterable[float],
    generator: GeneratorLike = None,
    round_fn: Callable[[float], int] = ...,
) -> list[list[int]]: ...
def balanced_monolabel_split(
    targets_indices: Tensor | list[int],
    num_classes: int,
    lengths: Iterable[float],
    generator: GeneratorLike = None,
    round_fn: Callable[[float], int] = ...,
) -> list[list[int]]: ...
