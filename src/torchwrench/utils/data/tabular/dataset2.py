#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import pandas as pd
import pythonwrench as pw


class TabularDatasetInterface:
    @abstractmethod
    def get_row(self, row_indexer) -> Any: ...

    @abstractmethod
    def get_column(self, col_indexer) -> Any: ...

    @property
    @abstractmethod
    def num_rows(self) -> int: ...

    @property
    @abstractmethod
    def num_columns(self) -> int: ...

    @property
    def shape(self) -> Tuple[int, int]:
        return self.num_rows, self.num_columns

    @property
    def ndim(self) -> int:
        return len(self.shape)


class DictListWrapper(TabularDatasetInterface):
    def __init__(self, data: Dict[Any, list]) -> None:
        super().__init__()
        self._data = data

    def get_row(self, row_indexer):
        return {k: v[row_indexer] for k, v in self._data.items()}

    def get_column(self, col_indexer):
        return self._data[col_indexer]

    @property
    def num_rows(self) -> int:
        if len(self._data) == 0:
            return 0
        else:
            return len(next(iter(self._data.values())))

    @property
    def num_columns(self) -> int:
        return len(self._data)


class ListDictWrapper(TabularDatasetInterface):
    def __init__(self, data: List[Dict]) -> None:
        super().__init__()
        self._data = data

    def get_row(self, row_indexer):
        return self._data[row_indexer]

    def get_column(self, col_indexer):
        return [data_i[col_indexer] for data_i in self._data]

    @property
    def num_rows(self) -> int:
        return len(self._data)

    @property
    def num_columns(self) -> int:
        if len(self._data) == 0:
            return 0
        else:
            return len(next(iter(self._data)))


class DataFrameWrapper(TabularDatasetInterface):
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__()
        self._data = data

    def get_row(self, row_indexer):
        return self._data[row_indexer]

    def get_column(self, col_indexer):
        return self._data[col_indexer]

    @property
    def num_rows(self) -> int:
        return self._data.shape[0]

    @property
    def num_columns(self) -> int:
        return self._data.shape[1]


class TabularDataset(TabularDatasetInterface):
    def __init__(self, data) -> None:
        if pw.isinstance_generic(data, Dict[Any, list]):
            wrapper = DictListWrapper(data)
        elif pw.isinstance_generic(data, List[Dict]):
            wrapper = ListDictWrapper(data)
        elif isinstance(data, pd.DataFrame):
            wrapper = DataFrameWrapper(data)
        else:
            msg = f"Invalid argument type {type(data)}."
            raise TypeError(msg)

        super().__init__()
        self._wrapper = wrapper

    def get_row(self, row_indexer):
        return self._wrapper.get_row(row_indexer)

    def get_column(self, col_indexer):
        return self._wrapper.get_column(col_indexer)

    @property
    def num_rows(self) -> int:
        return self._wrapper.num_rows

    @property
    def num_columns(self) -> int:
        return self._wrapper.num_columns
