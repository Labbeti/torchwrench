from io import TextIOBase
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, Mapping, MutableMapping

import pythonwrench as pw
from pythonwrench.typing import SupportsGetitemIterLen
from torch import Tensor
from typing_extensions import Self

from torchwrench.extras.numpy import np
from torchwrench.extras.pandas import pd
from torchwrench.extras.speechbrain import DynamicItemDataset

from ._core import (
    T_ColIndex as T_ColIndex,
)
from ._core import (
    T_RowIndex as T_RowIndex,
)
from ._core import (
    TabularDatasetInterface,
)

__all__ = ["TabularDataset", "T_RowIndex", "T_ColIndex"]

class TabularDataset(
    TabularDatasetInterface[T_RowIndex, T_ColIndex], Generic[T_RowIndex, T_ColIndex]
):
    def __init__(
        self,
        data: Mapping[Any, pw.SupportsGetitemIterLen]
        | pw.SupportsGetitemIterLen[dict[Any, Any]]
        | pd.DataFrame
        | Tensor
        | np.ndarray
        | DynamicItemDataset
        | TabularDatasetInterface[T_RowIndex, T_ColIndex],
        row_mapper: Mapping[T_RowIndex, T_RowIndex] | None = None,
        col_mapper: MutableMapping[T_ColIndex, T_ColIndex] | None = None,
        fns_list: Iterable[
            tuple[
                tuple[T_ColIndex, ...] | T_ColIndex,
                tuple[T_ColIndex, ...] | T_ColIndex,
                Callable,
            ]
        ] = (),
    ) -> None: ...
    @classmethod
    def from_csv(cls, fpath: str | Path | TextIOBase, **kwds) -> Self: ...
    @classmethod
    def from_json(cls, fpath: str | Path | TextIOBase, **kwds) -> Self: ...
    def add_dynamic_column(
        self,
        fn: Callable,
        requires: tuple[T_ColIndex, ...],
        provides: T_ColIndex | tuple[T_ColIndex, ...],
        add_to_output_keys: bool = True,
    ) -> None: ...
    @property
    def row_names(self) -> pw.SupportsGetitemIterLen: ...
    @property
    def column_names(self) -> SupportsGetitemIterLen: ...
    def to_dataframe(self) -> pd.DataFrame: ...
    def to_dict_list(self) -> dict[T_ColIndex, list]: ...
    def to_list_dict(self) -> list[dict[T_ColIndex, Any]]: ...
    def to_numpy(self) -> np.ndarray: ...
    def to_tensor(self) -> Tensor: ...
    def to_csv(self, fpath: str | Path, *args, **kwargs) -> None: ...
    def to_json(self, fpath: str | Path, *args, **kwargs) -> None: ...
    def __getitem__(self, indexer_) -> Any: ...
