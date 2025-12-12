#!/usr/bin/env python
# -*- coding: utf-8 -*-

from io import TextIOBase
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Tuple,
    Union,
)

import numpy as np
import pythonwrench as pw
from pythonwrench.typing import SupportsGetitemIterLen
from torch import Tensor
from typing_extensions import Self

import torchwrench as tw
from torchwrench.extras.pandas import pd
from torchwrench.extras.speechbrain import DynamicItemDataset
from torchwrench.serialization.csv import read_csv, save_csv
from torchwrench.serialization.json import read_json, save_json

from ._core import (
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
        row_mapper: Union[Mapping[T_RowIndex, T_RowIndex], None] = None,
        col_mapper: Union[MutableMapping[T_ColIndex, T_ColIndex], None] = None,
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
                wrapper,  # type: ignore
                fns_list,
            )

        super().__init__()
        self._wrapper = wrapper
        self._row_mapper = row_mapper
        self._col_mapper = col_mapper

    @property
    def _row_mapper_dict(self) -> Mapping[T_RowIndex, T_RowIndex]:
        if self._row_mapper is None:
            row_names = self._wrapper.row_names
            return dict(zip(row_names, row_names))  # type: ignore
        else:
            return self._row_mapper

    @property
    def _col_mapper_dict(self) -> MutableMapping[T_ColIndex, T_ColIndex]:
        if self._col_mapper is None:
            col_names = self._wrapper.column_names
            return dict(zip(col_names, col_names))
        else:
            return self._col_mapper

    @classmethod
    def from_csv(cls, fpath: Union[str, Path, TextIOBase], **kwds) -> Self:
        data = read_csv(fpath, **kwds)
        return cls(data)

    @classmethod
    def from_json(cls, fpath: Union[str, Path, TextIOBase], **kwds) -> Self:
        data = read_json(fpath, **kwds)
        return cls(data)

    def add_dynamic_column(
        self,
        fn: Callable,
        requires: Tuple[T_ColIndex, ...],
        provides: Union[T_ColIndex, Tuple[T_ColIndex, ...]],
        add_to_output_keys: bool = True,
    ) -> None:
        col_mapper_before = self._col_mapper_dict
        if isinstance(self._wrapper, FunctionWrapper):
            self._wrapper.add_dynamic_column(fn, requires, provides)
        else:
            self._wrapper = FunctionWrapper(
                self._wrapper,  # type: ignore
                [(requires, provides, fn)],
            )

        if add_to_output_keys and self._col_mapper is not None:
            if isinstance(provides, (str, int)):
                provides = [provides]  # type: ignore
            for provide in provides:  # type: ignore
                self._col_mapper[provide] = provide
        elif not add_to_output_keys and self._col_mapper is None:
            self._col_mapper = col_mapper_before

    @property
    def row_names(self) -> pw.SupportsGetitemIterLen:
        if self._row_mapper is None:
            return self._wrapper.row_names
        else:
            return tuple(self._row_mapper.keys())

    @property
    def column_names(self) -> SupportsGetitemIterLen:
        if self._col_mapper is None:
            return self._wrapper.column_names
        else:
            return tuple(self._col_mapper.keys())

    def to_dataframe(self) -> pd.DataFrame:
        list_dict = self.to_list_dict()
        return pd.DataFrame(list_dict)

    def to_dict_list(self) -> Dict[T_ColIndex, List]:
        list_dict = self.to_list_dict()
        return pw.list_dict_to_dict_list(list_dict, "same")

    def to_list_dict(self) -> List[Dict[T_ColIndex, Any]]:
        datalist = self._wrapper.to_list_dict()
        datalist = [
            {
                tgt_col: datalist[src_row][src_col]
                for tgt_col, src_col in self._col_mapper_dict.items()
            }
            for tgt_row, src_row in self._row_mapper_dict.items()
        ]
        return datalist

    def to_numpy(self) -> np.ndarray:
        datalist = self.to_list_dict()
        return np.array([list(item.values()) for item in datalist])

    def to_tensor(self) -> Tensor:
        datalist = self.to_list_dict()
        return tw.as_tensor([list(item.values()) for item in datalist])

    def to_csv(self, fpath: Union[str, Path], *args, **kwargs) -> None:
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
                self._row_mapper,  # type: ignore
                indexer.row,
            )

        if indexer.has_col_indexer:
            if self._col_mapper is None:
                col_indexer = indexer.col
            elif indexer.single_col:
                col_indexer = self._col_mapper[indexer.col]  # type: ignore
            else:
                col_indexer = [self._col_mapper[col] for col in indexer.col]  # type: ignore

            result = self._wrapper[row_indexer, col_indexer]  # type: ignore
        else:
            result = self._wrapper[row_indexer]  # type: ignore

        if self._col_mapper is None or indexer.has_col_indexer:
            pass
        elif indexer.single_row:
            result = {tgt: result[src] for tgt, src in self._col_mapper.items()}  # type: ignore
        else:
            result = [
                {tgt: result_i[src] for tgt, src in self._col_mapper.items()}  # type: ignore
                for result_i in result
            ]

        return result
