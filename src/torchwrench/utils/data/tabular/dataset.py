#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Tuple,
    Union,
)

import numpy as np
import pythonwrench as pw
import torch
from pythonwrench.typing import SupportsGetitemIterLen
from torch import Tensor

from torchwrench.extras.pandas import _PANDAS_AVAILABLE, pd
from torchwrench.extras.speechbrain import DynamicItemDataset
from torchwrench.serialization.csv import save_csv
from torchwrench.serialization.json import save_json

from ._core import (
    ColumnConcatWrapper,
    DataFrameWrapper,
    DictListWrapper,
    DynamicDatasetWrapper,
    FunctionWrapper,
    IndexerWrapper,
    T_ColIndex,
    T_RowIndex,
    TabularDatasetInterface,
    TensorOrArrayWrapper,
    _get_from_idx_indices_slice_mask,
)

__all__ = [
    "TabularDataset",
    "T_RowIndex",
    "T_ColIndex",
]


class TabularDataset(
    Generic[T_RowIndex, T_ColIndex],
    TabularDatasetInterface[T_RowIndex, T_ColIndex],
):
    def __init__(
        self,
        data: Union[
            Mapping[Any, pw.SupportsGetitemIterLen],
            pw.SupportsGetitemIterLen[Dict[Any, Any]],
            pd.DataFrame,
            Tensor,
            np.ndarray,
            DynamicItemDataset,
            TabularDatasetInterface[T_RowIndex, T_ColIndex],
        ],
        row_mapper: Union[SupportsGetitemIterLen, None] = None,
        col_mapper: Union[SupportsGetitemIterLen, None] = None,
        fns_list: Iterable[
            Tuple[
                Union[Tuple[T_ColIndex, ...], T_ColIndex],
                Union[Tuple[T_ColIndex, ...], T_ColIndex],
                Callable,
            ]
        ] = (),
    ) -> None:
        if isinstance(data, TabularDatasetInterface):
            wrapper = data
        elif pw.isinstance_generic(data, Mapping[Any, pw.SupportsGetitemIterLen]):
            wrapper = DictListWrapper(data)
        elif pw.isinstance_generic(data, pw.SupportsGetitemIterLen[Dict]):
            datadict = pw.list_dict_to_dict_list(data, "same")
            wrapper = DictListWrapper(datadict)
        elif isinstance(data, pd.DataFrame):
            wrapper = DataFrameWrapper(data)
        elif isinstance(data, (Tensor, np.ndarray)):
            wrapper = TensorOrArrayWrapper(data)
        elif isinstance(data, DynamicItemDataset):
            wrapper = DynamicDatasetWrapper(data)
        else:
            msg = f"Invalid argument type {type(data)}."
            raise TypeError(msg)

        fns_list = list(fns_list)
        if len(fns_list) > 0:
            wrapper = FunctionWrapper(
                self._wrapper,  # type: ignore
                fns_list,
            )

        super().__init__()
        self._wrapper = wrapper
        self._row_mapper = row_mapper
        self._col_mapper = col_mapper

    def concat_columns_with(
        self,
        other: TabularDatasetInterface[T_RowIndex, T_ColIndex],
    ) -> ColumnConcatWrapper[T_RowIndex, T_ColIndex]:
        return ColumnConcatWrapper([self, other])

    def add_dynamic_column(
        self,
        fn: Callable,
        requires: Tuple[T_ColIndex, ...],
        provides: Union[T_ColIndex, Tuple[T_ColIndex, ...]],
    ) -> None:
        if isinstance(self._wrapper, FunctionWrapper):
            self._wrapper.add_dynamic_column(fn, requires, provides)
        else:
            self._wrapper = FunctionWrapper(
                self._wrapper,  # type: ignore
                [(requires, provides, fn)],
            )

    @property
    def row_names(self) -> pw.SupportsGetitemIterLen:
        if self._row_mapper is None:
            return self._wrapper.row_names
        else:
            return range(len(self._row_mapper))

    @property
    def column_names(self) -> SupportsGetitemIterLen:
        if self._col_mapper is None:
            return self._wrapper.column_names
        else:
            return self._col_mapper

    def to_dataframe(self) -> pd.DataFrame:
        list_dict = self.to_list_dict()
        return pd.DataFrame(list_dict)

    def to_dict_list(self) -> Dict[T_ColIndex, List]:
        list_dict = self.to_list_dict()
        return pw.list_dict_to_dict_list(list_dict, "same")

    def to_list_dict(self) -> List[Dict[T_ColIndex, Any]]:
        datalist = self._wrapper.to_list_dict()
        datalist = [
            {col: datalist[row][col] for col in self.column_names}
            for row in self.row_names
        ]
        return datalist

    def to_numpy(self) -> np.ndarray:
        datalist = self.to_list_dict()
        return np.array([list(item.values()) for item in datalist])

    def to_tensor(self) -> Tensor:
        datalist = self.to_list_dict()
        return torch.as_tensor([list(item.values()) for item in datalist])

    def to_csv(self, fpath: Union[str, Path], *args, **kwargs) -> None:
        if _PANDAS_AVAILABLE:
            data = self.to_dataframe()
        else:
            data = self.to_dict_list()

        save_csv(data, fpath, *args, **kwargs)

    def to_json(self, fpath: Union[str, Path], *args, **kwargs) -> None:
        data = self.to_dict_list()
        save_json(data, fpath, *args, **kwargs)

    def __getitem__(self, indexer_, /) -> Any:
        indexer = IndexerWrapper(indexer_, self)
        del indexer_

        if self._row_mapper is None:
            row_indexer = indexer.row
        else:
            row_indexer = _get_from_idx_indices_slice_mask(
                self._row_mapper, indexer.row
            )

        if not indexer.has_col_indexer:
            result = self._wrapper[row_indexer]  # type: ignore
        else:
            if self._col_mapper is None:
                col_indexer = indexer.col
            elif indexer.single_col:
                col_indexer = self._col_mapper[indexer.col]  # type: ignore
            else:
                col_indexer = [self._col_mapper[col] for col in indexer.col]  # type: ignore

            result = self._wrapper[row_indexer, col_indexer]  # type: ignore

        return result
