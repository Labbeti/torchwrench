#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Mapping, Tuple, Union

import numpy as np
import pandas as pd
import pythonwrench as pw
from pythonwrench.typing import SupportsGetitemIterLen
from speechbrain.dataio.dataset import DynamicItemDataset
from torch import Tensor

from torchwrench.nn.functional.multilabel import multihot_to_multi_indices


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

    def keys(self) -> pw.SupportsGetitemIterLen:
        return self.column_names

    def values(self) -> list:
        return [self[k] for k in self.keys()]

    def __iter__(self):
        for row_index in range(self.num_rows):
            yield self[row_index]

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
    def to_dataframe(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def to_dict_list(self) -> Dict[Any, List]:
        raise NotImplementedError

    @abstractmethod
    def to_list_dict(self) -> List[Dict]:
        raise NotImplementedError

    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, indexer, /) -> Any:
        raise NotImplementedError


class DictListWrapper(TabularDatasetInterface):
    def __init__(self, data: Mapping[Any, pw.SupportsGetitemIterLen]) -> None:
        if not pw.all_eq(map(len, data.values())):
            msg = "Invalid data dict lengths."
            raise ValueError(msg)

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

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._data)

    def to_dict_list(self) -> Dict[Any, List]:
        return {k: list(v) for k, v in self._data.items()}

    def to_list_dict(self) -> List[Dict]:
        return pw.dict_list_to_list_dict(self._data, "same")

    def to_numpy(self) -> np.ndarray:
        return np.array(list(self._data.values())).T

    def __getitem__(self, indexer, /):
        has_col_indexer = True
        if pw.isinstance_generic(indexer, (int, slice, Iterable[int], Iterable[bool])):
            row_indexer = indexer
            col_indexer = self.column_names
            has_col_indexer = False
        elif isinstance(indexer, tuple) and len(indexer) == 2:
            row_indexer, col_indexer = indexer
        else:
            row_indexer = slice(None)
            col_indexer = indexer
        del indexer

        single = False
        if isinstance(col_indexer, (int, str)):
            col_indexer = [col_indexer]
            single = True

        if isinstance(row_indexer, int):
            result_dict = {col: self._data[col][row_indexer] for col in col_indexer}
            return result_dict
        elif isinstance(row_indexer, slice):
            result_dict = {col: self._data[col][row_indexer] for col in col_indexer}
        elif pw.isinstance_generic(row_indexer, Iterable[int]):
            result_dict = {
                col: _get_from_indices(self._data[col], row_indexer)
                for col in col_indexer
            }
        elif pw.isinstance_generic(row_indexer, Iterable[bool]):
            result_dict = {
                col: _get_from_mask(self._data[col], row_indexer) for col in col_indexer
            }
        else:
            msg = f"Invalid argument type {type(row_indexer)}."
            raise TypeError(msg)

        if has_col_indexer:
            if single:
                return next(iter(result_dict.values()))
            else:
                return list(result_dict.values())
        else:
            result_list = pw.dict_list_to_list_dict(result_dict, "same")
            if single:
                return result_list[0]
            else:
                return result_list


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

    def to_dataframe(self) -> pd.DataFrame:
        return self._data

    def to_dict_list(self) -> dict:
        return self._data.to_dict("list")

    def to_list_dict(self) -> list:
        return self._data.to_dict("records")

    def to_numpy(self) -> np.ndarray:
        return self._data.to_numpy()

    def __getitem__(self, indexer, /):
        return self._data[indexer]


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

    def to_dataframe(self) -> pd.DataFrame:
        data = self._data
        if isinstance(data, Tensor):
            data = data.numpy(force=True)
        return pd.DataFrame(data)

    def to_dict_list(self) -> Dict[Any, List]:
        return dict(zip(self.column_names, map(list, self._data.transpose(0, 1))))

    def to_dynamic_item_dataset(self, id_column_name: int = 0) -> DynamicItemDataset:
        data_ids = self._data[id_column_name]
        data = {id_: self._data[i] for i, id_ in enumerate(data_ids)}
        dset = DynamicItemDataset(data)
        return dset

    def to_list_dict(self) -> List[dict]:
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
        return pd.DataFrame(self[:])

    def to_dict_list(self) -> Dict[Any, List]:
        datalist = self[:]
        return pw.list_dict_to_dict_list(datalist)  # type: ignore

    def to_list_dict(self) -> List[Dict]:
        datalist = self[:]
        return datalist  # type: ignore

    def to_numpy(self) -> np.ndarray:
        datalist = self[:]
        return np.array([list(item.values()) for item in datalist])  # type: ignore

    def __getitem__(self, indexer, /):
        if isinstance(indexer, int):
            return self._data[indexer]

        elif isinstance(indexer, slice):
            datalist = [self._data[idx] for idx in range(len(self))[indexer]]
            return datalist

        elif pw.isinstance_generic(indexer, Iterable[int]):
            datalist = [self._data[idx] for idx in indexer]
            return datalist

        elif pw.isinstance_generic(indexer, Iterable[bool]):
            indices = multihot_to_multi_indices(indexer)
            datalist = [self._data[idx] for idx in indices]
            return datalist

        elif isinstance(indexer, tuple) and len(indexer) == 2:
            row_indexer, col_indexer = indexer
        else:
            row_indexer = slice(None)
            col_indexer = indexer
        del indexer

        single = False
        if isinstance(col_indexer, (int, str)):
            col_indexer = [col_indexer]
            single = True

        with self._data.output_keys_as(col_indexer):
            if isinstance(row_indexer, int):
                result = self._data[row_indexer]

            elif isinstance(row_indexer, slice):
                datalist = [self._data[idx] for idx in range(len(self))[row_indexer]]
                result = list(pw.list_dict_to_dict_list(datalist, "same").values())

            elif pw.isinstance_generic(row_indexer, Iterable[int]):
                datalist = [self._data[idx] for idx in row_indexer]
                result = list(pw.list_dict_to_dict_list(datalist, "same").values())

            elif pw.isinstance_generic(row_indexer, Iterable[bool]):
                indices = multihot_to_multi_indices(row_indexer)
                datalist = [self._data[idx] for idx in indices]
                result = list(pw.list_dict_to_dict_list(datalist, "same").values())

            else:
                raise TypeError

            if single:
                return result[0]
            else:
                return result


class TabularDataset(TabularDatasetInterface):
    def __init__(
        self,
        data: Union[
            Mapping[Any, pw.SupportsGetitemIterLen],
            pw.SupportsGetitemIterLen[Dict[Any, Any]],
            pd.DataFrame,
            Tensor,
            np.ndarray,
            DynamicItemDataset,
        ],
        row_mapper: Union[SupportsGetitemIterLen, None] = None,
        column_mapper: Union[SupportsGetitemIterLen, None] = None,
    ) -> None:
        if pw.isinstance_generic(data, Mapping[Any, pw.SupportsGetitemIterLen]):
            wrapper = DictListWrapper(data)
        elif pw.isinstance_generic(data, List[Dict]):
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

    def to_dataframe(self) -> pd.DataFrame:
        return self._wrapper.to_dataframe()

    def to_dict_list(self) -> Dict[Any, List]:
        return self._wrapper.to_dict_list()

    def to_list_dict(self) -> List[Dict]:
        return self._wrapper.to_list_dict()

    def to_numpy(self) -> np.ndarray:
        return self._wrapper.to_numpy()

    def __getitem__(self, indexer, /) -> Any:
        if isinstance(indexer, (int, slice)):
            if self._row_mapper is not None:
                indexer = self._row_mapper[indexer]
            return self._wrapper[indexer]

        elif pw.isinstance_generic(indexer, Iterable[int]):
            if self._row_mapper is not None:
                indexer = _get_from_indices(self._row_mapper, indexer)
            return self._wrapper[indexer]

        elif pw.isinstance_generic(indexer, Iterable[bool]):
            indices = multihot_to_multi_indices(indexer)
            return self[indices]

        elif isinstance(indexer, tuple) and len(indexer) == 2:
            row_indexer, col_indexer = indexer
        else:
            row_indexer = slice(None)
            col_indexer = indexer
        del indexer

        if self._row_mapper is not None:
            if isinstance(row_indexer, (int, slice)):
                row_indexer = self._row_mapper[row_indexer]
            elif pw.isinstance_generic(row_indexer, Iterable[int]):
                row_indexer = _get_from_indices(self._row_mapper, row_indexer)
            elif pw.isinstance_generic(row_indexer, Iterable[bool]):
                row_indexer = _get_from_mask(self._row_mapper, row_indexer)
            else:
                raise TypeError

        single = False
        if isinstance(col_indexer, (int, str)):
            col_indexer = [col_indexer]
            single = True

        if self._col_mapper is not None:
            col_indexer = [self._col_mapper[col] for col in col_indexer]

        result = self._wrapper[row_indexer, col_indexer]
        if single:
            return result[0]
        else:
            return result


def _get_from_indices(
    x: Union[Tensor, np.ndarray, Iterable],
    indices: Union[Tensor, np.ndarray, Iterable],
) -> Union[Tensor, np.ndarray, list]:
    if isinstance(x, Tensor) and isinstance(indices, Tensor):
        return x[indices]
    if isinstance(x, np.ndarray) and isinstance(indices, np.ndarray):
        return x[indices]
    if isinstance(x, list):
        return [x[idx] for idx in indices]

    raise TypeError


def _get_from_mask(
    x: Union[Tensor, np.ndarray, Iterable],
    mask: Union[Tensor, np.ndarray, Iterable],
) -> Union[Tensor, np.ndarray, list]:
    if isinstance(x, Tensor) and isinstance(mask, Tensor):
        return x[mask]
    if isinstance(x, np.ndarray) and isinstance(mask, np.ndarray):
        return x[mask]
    if isinstance(x, list):
        indices = multihot_to_multi_indices(mask)
        return [x[idx] for idx in indices]

    raise TypeError
