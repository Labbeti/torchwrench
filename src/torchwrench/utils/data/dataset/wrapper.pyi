import abc
from abc import abstractmethod
from typing import Any, Callable, Generic, Iterable, Iterator, TypeVar

from _typeshed import Incomplete
from pythonwrench.typing.classes import (
    SupportsGetitemIterLen,
    SupportsGetitemLen,
)
from pythonwrench.typing.classes import (
    SupportsIterLen as SupportsIterLen,
)
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.dataset import Subset as TorchSubset

from torchwrench.types.tensor_subclasses import LongTensor1D as LongTensor1D

T = TypeVar("T", covariant=True)
U = TypeVar("U", covariant=True)
SizedDatasetLike = SupportsGetitemLen
T_Dataset = TypeVar("T_Dataset", bound=Dataset)
T_SizedDatasetLike = TypeVar("T_SizedDatasetLike", bound=SupportsGetitemLen)
T_SupportsIterLenDataset = TypeVar(
    "T_SupportsIterLenDataset", bound=SupportsGetitemIterLen
)

class EmptyDataset(Dataset[None]):
    def __getitem__(self, idx) -> None: ...
    def __len__(self) -> int: ...

class _WrapperBase(Dataset[T], Generic[T], metaclass=abc.ABCMeta):
    dataset: Incomplete
    def __init__(self, dataset: Any) -> None: ...
    @abstractmethod
    def __len__(self) -> int: ...
    def unwrap(self, recursive: bool = True) -> SupportsGetitemLen | Dataset: ...

class Wrapper(_WrapperBase[T], Generic[T], metaclass=abc.ABCMeta):
    @abstractmethod
    def __getitem__(self, idx) -> T: ...

class IterableWrapper(
    IterableDataset[T], _WrapperBase[T], Generic[T], metaclass=abc.ABCMeta
):
    dataset: SupportsGetitemLen[T] | SupportsIterLen[T]
    def __init__(self, dataset: SupportsGetitemLen[T] | SupportsIterLen[T]) -> None: ...
    @abstractmethod
    def __iter__(self) -> Iterator[T]: ...

class TransformWrapper(Wrapper[T], Generic[T, U]):
    dataset: SupportsGetitemLen[T]
    def __init__(
        self,
        dataset: SupportsGetitemLen[T],
        transform: Callable[[T], U] | None,
        condition: Callable[[T, int], bool] | None = None,
    ) -> None: ...
    def __getitem__(self, idx) -> T | U: ...
    def __len__(self) -> int: ...
    @property
    def transform(self) -> Callable[[T], U] | None: ...
    @property
    def condition(self) -> Callable[[T, int], bool] | None: ...

class IterableTransformWrapper(IterableWrapper[T], Generic[T, U]):
    def __init__(
        self,
        dataset: SupportsGetitemLen[T] | SupportsIterLen[T],
        transform: Callable[[T], U] | None,
        condition: Callable[[T, int], bool] | None = None,
    ) -> None: ...
    def __iter__(self) -> Iterator[T | U]: ...
    def __len__(self) -> int: ...
    @property
    def transform(self) -> Callable[[T], U] | None: ...
    @property
    def condition(self) -> Callable[[T, int], bool] | None: ...

class IterableSubset(IterableWrapper[T], Generic[T]):
    def __init__(
        self,
        dataset: SupportsGetitemLen[T] | SupportsIterLen[T],
        indices: Iterable[int] | LongTensor1D,
    ) -> None: ...
    def __iter__(self) -> Iterator[T]: ...
    def __len__(self) -> int: ...

class Subset(TorchSubset[T], Wrapper[T], Generic[T], metaclass=abc.ABCMeta):
    def __init__(
        self, dataset: SizedDatasetLike[T], indices: Iterable[int]
    ) -> None: ...
