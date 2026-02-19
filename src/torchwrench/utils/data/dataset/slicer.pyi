import abc
from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, TypeVar, overload

from pythonwrench.typing.classes import SupportsGetitemLen as SupportsGetitemLen
from torch.utils.data.dataset import Dataset
from typing_extensions import TypeAlias

from torchwrench.extras.numpy import np as np
from torchwrench.extras.numpy.functional import (
    is_numpy_bool_array as is_numpy_bool_array,
)
from torchwrench.nn.functional.transform import as_tensor as as_tensor
from torchwrench.types._typing import BoolTensor1D as BoolTensor1D
from torchwrench.types._typing import Tensor1D as Tensor1D
from torchwrench.types._typing import TensorOrArray as TensorOrArray
from torchwrench.types.guards import is_number_like as is_number_like
from torchwrench.types.guards import is_tensor_or_array as is_tensor_or_array
from torchwrench.utils.data.dataset.wrapper import Wrapper as Wrapper

T = TypeVar("T", covariant=True)
U = TypeVar("U", covariant=True)
MultiIndexer: TypeAlias

class DatasetSlicer(ABC, Dataset[T], Generic[T], metaclass=abc.ABCMeta):
    def __init__(
        self,
        *,
        add_slice_support: bool = True,
        add_indices_support: bool = True,
        add_mask_support: bool = True,
        add_none_support: bool = True,
    ) -> None: ...
    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def get_item(self, idx, /, *args, **kwargs) -> Any: ...
    @overload
    def __getitem__(self, idx: int) -> T: ...
    @overload
    def __getitem__(self, idx: MultiIndexer) -> list[T]: ...
    @overload
    def __getitem__(self, idx: tuple[Any, ...]) -> Any: ...
    def get_items_indices(
        self, indices: Iterable[int] | TensorOrArray, *args
    ) -> list[T]: ...
    def get_items_mask(
        self, mask: Iterable[bool] | TensorOrArray, *args
    ) -> list[T]: ...
    def get_items_slice(self, slice_: slice, *args) -> list[T]: ...
    def get_items_none(self, none: None, *args) -> list[T]: ...

class DatasetSlicerWrapper(DatasetSlicer[T], Wrapper[T], Generic[T]):
    def __init__(
        self,
        dataset: SupportsGetitemLen[T],
        *,
        add_slice_support: bool = True,
        add_indices_support: bool = True,
        add_mask_support: bool = True,
        add_none_support: bool = True,
    ) -> None: ...
    def __len__(self) -> int: ...
    def get_item(self, idx: int, *args) -> T: ...
