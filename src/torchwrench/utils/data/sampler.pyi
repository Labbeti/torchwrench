from typing import Iterable, Iterator, Literal, Sequence

from torch import Tensor as Tensor
from torch.utils.data.sampler import Sampler

from torchwrench.nn.functional.transform import GeneratorLike as GeneratorLike
from torchwrench.nn.functional.transform import as_generator as as_generator
from torchwrench.nn.functional.transform import as_tensor as as_tensor
from torchwrench.nn.functional.transform import shuffled as shuffled

class SubsetSampler(Sampler[int]):
    def __init__(self, indices: list[int] | Tensor) -> None: ...
    def __iter__(self) -> Iterator[int]: ...
    def __len__(self) -> int: ...

class SubsetCycleSampler(Sampler[int]):
    def __init__(
        self,
        indices: Tensor | Iterable[int],
        n_max_iterations: int | Literal["inf"] = "inf",
        shuffle: bool = True,
        seed: GeneratorLike = None,
    ) -> None: ...
    def __iter__(self) -> Iterator[int]: ...
    def __len__(self) -> int: ...

class BalancedSampler(Sampler):
    def __init__(
        self,
        indices_per_class: Sequence[Sequence[int]],
        n_max_iterations: int,
        shuffle: bool = True,
        seed: GeneratorLike = None,
    ) -> None: ...
    def __iter__(self) -> Iterator[int]: ...
    def __len__(self) -> int: ...
