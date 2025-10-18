#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple, Union

import numpy as np
import pandas as pd
import pythonwrench as pw
from pythonwrench.typing import SupportsGetitemIterLen
from speechbrain.dataio.dataset import DynamicItemDataset
from torch import Tensor

from torchwrench.nn.functional.multilabel import multihot_to_multi_indices


class TabularIndexer:
    def __init__(self, indexer) -> None:
        super().__init__()

        if pw.isinstance_generic(indexer, (int, slice, Iterable[int], Iterable[bool])):
            row_indexer = indexer
            col_indexer = None
            has_col_indexer = False
        elif isinstance(indexer, tuple) and len(indexer) == 2:
            row_indexer, col_indexer = indexer
            has_col_indexer = True
        else:
            row_indexer = slice(None)
            col_indexer = indexer
            has_col_indexer = True

        self._row_indexer = row_indexer
        self._col_indexer = col_indexer
        self._has_col_indexer = has_col_indexer

    @property
    def row(self) -> Any:
        return self._row_indexer

    @property
    def col(self) -> Any:
        return self._col_indexer

    @property
    def has_col_indexer(self) -> Any:
        return self._has_col_indexer

    @property
    def single_row(self) -> Any:
        return isinstance(self._row_indexer, (int, str))

    @property
    def single_col(self) -> Any:
        return isinstance(self._col_indexer, (int, str))


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
        lengths = list(map(len, data.values()))
        if not pw.all_eq(lengths):
            msg = f"Invalid data dict lengths. (found {lengths})"
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
        row_indexer, col_indexer, has_col_indexer = _get_row_col_indexer(indexer)
        del indexer

        if not has_col_indexer:
            col_indexer = self.column_names

        single_col = False
        if isinstance(col_indexer, (int, str)):
            col_indexer = [col_indexer]
            single_col = True

        single_row = False
        if isinstance(row_indexer, int):
            result_dict = {col: self._data[col][row_indexer] for col in col_indexer}
            single_row = True
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
            if single_col:
                return next(iter(result_dict.values()))
            else:
                return list(result_dict.values())
        else:
            if single_col:
                return next(iter(result_dict.values()))
            elif single_row:
                return result_dict
            else:
                result_list = pw.dict_list_to_list_dict(result_dict, "same")
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
        row_indexer, col_indexer, _ = _get_row_col_indexer(indexer)
        del indexer

        single_row = False
        single_col = False

        if isinstance(col_indexer, (int, str)):
            col_indexer = [col_indexer]
            single_col = True

        if col_indexer is None:
            output_keys = list(self._data.pipeline.output_mapping.keys())
        else:
            output_keys = col_indexer

        with self._data.output_keys_as(output_keys):
            if isinstance(row_indexer, int):
                result = self._data[row_indexer]
                single_row = True

            elif isinstance(row_indexer, slice):
                result = [self._data[idx] for idx in range(len(self))[row_indexer]]

            elif pw.isinstance_generic(row_indexer, Iterable[int]):
                result = [self._data[idx] for idx in row_indexer]

            elif pw.isinstance_generic(row_indexer, Iterable[bool]):
                indices = multihot_to_multi_indices(row_indexer)
                result = [self._data[idx] for idx in indices]

            else:
                raise TypeError

        if col_indexer is None:
            return result
        elif single_row and single_col:
            return result[col_indexer[0]]  # type: ignore
        elif single_row and not single_col:
            return {col: result[col] for col in col_indexer}  # type: ignore
        elif not single_row and single_col:
            return [result_i[col_indexer[0]] for result_i in result]
        else:
            return [[result_i[col] for result_i in result] for col in col_indexer]


class FunctionWrapper(TabularDatasetInterface):
    def __init__(
        self,
        ds: TabularDatasetInterface,
        fns_list: Iterable[Tuple[Tuple[str, ...], Tuple[str, ...], Callable]],
        size: int,
    ) -> None:
        fns = {
            tuple(provides): (tuple(requires), fn)
            for requires, provides, fn in fns_list
        }
        super().__init__()
        self._ds = ds
        self._fns = fns
        self._size = size

    @property
    def row_names(self) -> range:
        return range(self._size)

    @property
    def column_names(self) -> tuple:
        return tuple(col for provides in self._fns.keys() for col in provides)

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

    def __getitem__(self, indexer, /) -> Any:
        row_indexer, col_indexer, _ = _get_row_col_indexer(indexer)
        del indexer

        if col_indexer is None:
            outputs_keys = self.column_names
        else:
            outputs_keys = col_indexer

        if isinstance(row_indexer, int):
            result = self._get_values(row_indexer, outputs_keys)
        elif isinstance(row_indexer, slice):
            row_indexer = range(len(self))[row_indexer]
            result = [self._get_values(idx, outputs_keys) for idx in row_indexer]
        elif pw.isinstance_generic(row_indexer, Iterable[int]):
            result = [self._get_values(idx, outputs_keys) for idx in row_indexer]
        elif pw.isinstance_generic(row_indexer, Iterable[bool]):
            row_indexer = multihot_to_multi_indices(row_indexer)
            result = [self._get_values(idx, outputs_keys) for idx in row_indexer]
        else:
            raise TypeError

        if col_indexer is None:
            return result
        elif isinstance(row_indexer, int):
            return [result[col] for col in col_indexer]
        elif isinstance(col_indexer, (int, str)):
            return [result_i[col_indexer] for result_i in result]  # type: ignore
        else:
            return [[result_i[col] for col in col_indexer] for result_i in result]

    def _get_values(self, idx: int, columns: Iterable[str]) -> Dict[str, Any]:
        result = {}
        for col in columns:
            if col in result:
                continue

            if col in self._ds.column_names:
                result |= {col: self._ds[idx, col]}
                continue

            for provides, (requires, fn) in self._fns.items():
                if col not in provides:
                    continue

                fn_inputs_dict = self._get_values(idx, requires)
                if not isinstance(requires, (tuple, list)):
                    fn_inputs = [fn_inputs_dict[requires]]
                else:
                    fn_inputs = [fn_inputs_dict[k] for k in requires]

                fn_output = fn(*fn_inputs)

                if not isinstance(provides, (tuple, list)):
                    fn_output_dict = {provides: fn_output}
                else:
                    fn_output_dict = dict(zip(provides, fn_output))

                result |= fn_inputs_dict | fn_output_dict
                break
        return result


class ColumnConcatWrapper(TabularDatasetInterface):
    def __init__(self, tabulars: Iterable[TabularDatasetInterface]) -> None:
        tabulars = list(tabulars)
        super().__init__()
        self._tabulars = tabulars

    @property
    def row_names(self) -> SupportsGetitemIterLen:
        if len(self._tabulars) == 0:
            return []
        return self._tabulars[0].row_names

    @property
    def column_names(self) -> SupportsGetitemIterLen:
        return [col for tabular in self._tabulars for col in tabular.column_names]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.concat(
            [tabular.to_dataframe() for tabular in self._tabulars], axis="columns"
        )

    def to_dict_list(self) -> Dict[Any, List]:
        return pw.reduce_or(
            [tabular.to_dict_list() for tabular in self._tabulars], start={}
        )

    def to_list_dict(self) -> List[Dict]:
        return pw.dict_list_to_list_dict(self.to_dict_list())

    def to_numpy(self) -> np.ndarray:
        return np.concat([tabular.to_numpy() for tabular in self._tabulars], axis=0)

    def __getitem__(self, indexer, /) -> Any:
        row_indexer, col_indexer, _ = _get_row_col_indexer(indexer)
        del indexer

        if col_indexer is None:
            result = [tabular[row_indexer] for tabular in self._tabulars]
        elif isinstance(col_indexer, str):
            result = [
                tabular[row_indexer, col_indexer]
                for tabular in self._tabulars
                if col_indexer in tabular.column_names
            ]
            result = result[0]
            return result
        elif pw.isinstance_generic(col_indexer, Iterable[str]):
            result = []
            for tabular in self._tabulars:
                included = [col for col in col_indexer if col in tabular.column_names]
                if len(included) == 0:
                    continue
                result_i = tabular[row_indexer, included]
                result.append(result_i)

            if isinstance(row_indexer, int):
                return pw.reduce_or(result, start={})
            else:
                result_2: list[dict] = []
                for i, result_lst_dic in enumerate(result):
                    item = pw.reduce_or(
                        [result[i][j] for j in range(len(result_lst_dic))], start={}
                    )
                    result_2.append(item)
                return result_2


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
        col_mapper: Union[SupportsGetitemIterLen, None] = None,
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
        self._col_mapper = col_mapper

    def add_dynamic_column(
        self,
        fn: Callable,
        requires: Tuple[str, ...],
        provides: Tuple[str, ...],
    ) -> None:
        self._wrapper = FunctionWrapper(
            self._wrapper, [(requires, provides, fn)], self.num_rows
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
        return self._wrapper.to_dataframe()

    def to_dict_list(self) -> Dict[Any, List]:
        return self._wrapper.to_dict_list()

    def to_list_dict(self) -> List[Dict]:
        return self._wrapper.to_list_dict()

    def to_numpy(self) -> np.ndarray:
        return self._wrapper.to_numpy()

    def __getitem__(self, indexer, /) -> Any:
        row_indexer, col_indexer, _ = _get_row_col_indexer(indexer)
        del indexer

        if self._row_mapper is not None:
            if isinstance(row_indexer, int):
                row_indexer = self._row_mapper[row_indexer]
            elif isinstance(row_indexer, slice):
                row_indexer = self._row_mapper[row_indexer]
            elif pw.isinstance_generic(row_indexer, Iterable[int]):
                row_indexer = _get_from_indices(self._row_mapper, row_indexer)
            elif pw.isinstance_generic(row_indexer, Iterable[bool]):
                row_indexer = _get_from_mask(self._row_mapper, row_indexer)
            else:
                raise TypeError

        if col_indexer is None:
            result = self._wrapper[row_indexer]
        else:
            if self._col_mapper is None:
                pass
            elif isinstance(col_indexer, (int, str)):
                col_indexer = self._col_mapper[col_indexer]
            else:
                col_indexer = [self._col_mapper[col] for col in col_indexer]

            result = self._wrapper[row_indexer, col_indexer]

        return result


def _get_row_col_indexer(
    indexer: Any,
) -> Tuple[
    Union[int, slice, Iterable[int], Iterable[bool]],
    Union[int, str, Iterable[int], Iterable[str], None],
    bool,
]:
    if pw.isinstance_generic(indexer, (int, slice, Iterable[int], Iterable[bool])):
        row_indexer = indexer
        col_indexer = None
        has_col_indexer = False
    elif isinstance(indexer, tuple) and len(indexer) == 2:
        row_indexer, col_indexer = indexer
        has_col_indexer = True
    else:
        row_indexer = slice(None)
        col_indexer = indexer
        has_col_indexer = True

    return row_indexer, col_indexer, has_col_indexer


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
