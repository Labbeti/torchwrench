#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import pythonwrench as pw
from pythonwrench.typing import SupportsGetitemIterLen
from speechbrain.dataio.dataset import DynamicItemDataset
from torch import Tensor


class TabularDatasetInterface:
    @property
    def num_rows(self) -> int:
        return len(self.row_names)

    @property
    def num_columns(self) -> int:
        return len(self.column_names)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.num_rows, self.num_columns

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def keys(self) -> SupportsGetitemIterLen:
        return self.column_names

    def values(self) -> Iterable:
        return [self.get_column(col_index) for col_index in self.column_names]

    def __iter__(self):
        for row_index in range(self.num_rows):
            yield self.get_row(row_index)

    def __len__(self) -> int:
        return self.num_rows

    @property
    @abstractmethod
    def row_names(self) -> SupportsGetitemIterLen:
        raise NotImplementedError

    @property
    @abstractmethod
    def column_names(self) -> SupportsGetitemIterLen:
        raise NotImplementedError

    @abstractmethod
    def get_row(self, row_indexer) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_column(self, col_indexer) -> Any:
        raise NotImplementedError

    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def to_dict_list(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def to_list_dict(self) -> list:
        raise NotImplementedError

    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, indexer, /) -> Any:
        raise NotImplementedError


class DictListWrapper(TabularDatasetInterface):
    def __init__(self, data: Dict[Any, list]) -> None:
        super().__init__()
        self._data = data

    @property
    def row_names(self) -> range:
        if len(self._data) == 0:
            return range(0)
        else:
            return range(len(next(iter(self._data.values()))))

    @property
    def column_names(self) -> tuple:
        return tuple(self._data.keys())

    def get_row(self, row_indexer):
        return {k: v[row_indexer] for k, v in self._data.items()}

    def get_column(self, col_indexer):
        return self._data[col_indexer]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._data)

    def to_dict_list(self) -> dict:
        return self._data

    def to_list_dict(self) -> list:
        return pw.dict_list_to_list_dict(self._data, "same")

    def to_numpy(self) -> np.ndarray:
        return np.array(list(self.values()))

    def __getitem__(self, indexer, /):
        if isinstance(indexer, tuple) and len(indexer) == 2:
            row_indexer, col_indexer = indexer
        else:
            row_indexer = indexer
            col_indexer = None
        del indexer

        # {k: v[row_indexer] for k, v in self._data.items()}

        # if isinstance(indexer, int):
        #     return self.get_row(indexer)
        # elif isinstance(row_indexer, slice):
        #     data = {k: self._data[k][row_indexer] for k in col_indexer}
        #     return pw.dict_list_to_list_dict(data, "same")
        # elif pw.isinstance_generic(indexer, Iterable[int]):
        #     return [self.get_row(row_index) for row_index in indexer]
        # else:
        #     raise TypeError
        raise NotImplementedError


class ListDictWrapper(TabularDatasetInterface):
    def __init__(self, data: List[Dict]) -> None:
        super().__init__()
        self._data = data

    @property
    def row_names(self) -> range:
        return range(len(self._data))

    @property
    def column_names(self) -> tuple:
        if len(self._data) == 0:
            return ()
        else:
            return tuple(next(iter(self._data[0].keys())))

    def get_row(self, row_indexer):
        return self._data[row_indexer]

    def get_column(self, col_indexer):
        return [data_i[col_indexer] for data_i in self._data]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._data)

    def to_dict_list(self) -> dict:
        return pw.list_dict_to_dict_list(self._data, "same")

    def to_list_dict(self) -> list:
        return self._data

    def to_numpy(self) -> np.ndarray:
        return np.array([data_i.values() for data_i in self._data])

    def __getitem__(self, indexer, /):
        if isinstance(indexer, int):
            return self.get_row(indexer)
        elif isinstance(indexer, slice):
            return self._data[indexer]
        elif pw.isinstance_generic(indexer, Iterable[int]):
            return [self.get_row(row_index) for row_index in indexer]
        else:
            raise TypeError


class DataFrameWrapper(TabularDatasetInterface):
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__()
        self._data = data

    @property
    def row_names(self) -> range:
        return range(self._data.shape[0])

    @property
    def column_names(self) -> tuple:
        return tuple(self._data.keys())

    def get_row(self, row_indexer):
        return self._data[row_indexer]

    def get_column(self, col_indexer):
        return self._data[col_indexer]

    def to_dataframe(self) -> pd.DataFrame:
        return self._data

    def to_dict_list(self) -> dict:
        return self._data.to_dict("list")

    def to_list_dict(self) -> list:
        return self._data.to_dict("records")

    def to_numpy(self) -> np.ndarray:
        return self._data.to_numpy()

    def __getitem__(self, indexer, /):
        if isinstance(indexer, int):
            return self.get_row(indexer)
        elif isinstance(indexer, slice):
            return self._data[indexer]
        elif pw.isinstance_generic(indexer, Iterable[int]):
            return [self.get_row(row_index) for row_index in indexer]
        else:
            raise TypeError


class TensorOrArrayWrapper(TabularDatasetInterface):
    def __init__(self, data: Union[Tensor, np.ndarray]) -> None:
        if data.ndim < 2:
            msg = f"Invalid number of dimensions for TabularDataset. (expected at least 2 dims but found {data.ndim})"
            raise ValueError(msg)
        super().__init__()
        self._data = data

    @property
    def row_names(self) -> range:
        return range(self._data.shape[0])

    @property
    def column_names(self) -> range:
        return range(self._data.shape[1])

    def get_row(self, row_indexer):
        return self._data[row_indexer]

    def get_column(self, col_indexer):
        return self._data[:, col_indexer]

    def to_dataframe(self) -> pd.DataFrame:
        data = self._data
        if isinstance(data, Tensor):
            data = data.numpy(force=True)
        return pd.DataFrame(data)

    def to_dict_list(self) -> dict:
        return dict(zip(self.column_names, map(list, self._data.transpose(0, 1))))

    def to_list_dict(self) -> list:
        return [dict(zip(self.column_names, data_i)) for data_i in self._data]

    def to_numpy(self) -> np.ndarray:
        data = self._data
        if isinstance(data, Tensor):
            data = data.numpy(force=True)
        return data

    def __getitem__(self, indexer, /):
        return self._data[indexer]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape


class DynamicDatasetWrapper(TabularDatasetInterface):
    def __init__(self, data: DynamicItemDataset) -> None:
        super().__init__()
        self._data = data

    @property
    def row_names(self) -> range:
        return range(len(self._data))

    @property
    def column_names(self) -> tuple:
        return tuple(self._data.pipeline.output_mapping)

    def get_row(self, row_indexer):
        return self._data[row_indexer]

    def get_column(self, col_indexer):
        with self._data.output_keys_as(col_indexer):
            return [data_i[col_indexer] for data_i in self._data]

    def to_dataframe(self) -> pd.DataFrame:
        raise NotImplementedError

    def to_dict_list(self) -> dict:
        raise NotImplementedError

    def to_list_dict(self) -> list:
        raise NotImplementedError

    def to_numpy(self) -> np.ndarray:
        return np.array([self._data[row_index] for row_index in self.row_names])

    def __getitem__(self, indexer, /):
        if isinstance(indexer, int):
            return self.get_row(indexer)
        elif isinstance(indexer, slice):
            return [self.get_row(row_index) for row_index in self.row_names[indexer]]
        elif pw.isinstance_generic(indexer, Iterable[int]):
            return [self.get_row(row_index) for row_index in indexer]
        else:
            raise TypeError


class TabularDataset(TabularDatasetInterface):
    def __init__(
        self,
        data: Union[
            Dict[Any, list],
            List[Dict],
            pd.DataFrame,
            np.ndarray,
            DynamicItemDataset,
        ],
        row_mapper: Union[SupportsGetitemIterLen, None] = None,
        column_mapper: Union[SupportsGetitemIterLen, None] = None,
    ) -> None:
        if pw.isinstance_generic(data, Dict[Any, list]):
            wrapper = DictListWrapper(data)
        elif pw.isinstance_generic(data, List[Dict]):
            wrapper = ListDictWrapper(data)
        elif isinstance(data, pd.DataFrame):
            wrapper = DataFrameWrapper(data)
        elif isinstance(data, (Tensor, np.ndarray)):
            wrapper = TensorOrArrayWrapper(data)
        elif isinstance(data, DynamicItemDataset):
            wrapper = DynamicDatasetWrapper(data)
        else:
            msg = f"Invalid argument type {type(data)}."
            raise TypeError(msg)

        super().__init__()
        self._wrapper = wrapper
        self._row_mapper = row_mapper
        self._col_mapper = column_mapper

    @property
    def row_names(self) -> range:
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

    def get_row(self, row_indexer):
        if self._row_mapper is not None:
            row_indexer = self._row_mapper[row_indexer]
        return self._wrapper.get_row(row_indexer)

    def get_column(self, col_indexer):
        if self._col_mapper is not None:
            col_indexer = self._col_mapper[col_indexer]
        return self._wrapper.get_column(col_indexer)

    def to_dict_list(self) -> dict:
        return self._wrapper.to_dict_list()

    def to_list_dict(self) -> list:
        return self._wrapper.to_list_dict()

    def to_numpy(self) -> np.ndarray:
        return self._wrapper.to_numpy()

    def __getitem__(self, indexer, /) -> Any:
        return self._wrapper[indexer]
