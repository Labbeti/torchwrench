#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Mapping,
    Tuple,
    Union,
    overload,
)

import numpy as np
import pythonwrench as pw
import torch
from pythonwrench.typing import SupportsGetitemIterLen
from torch import Tensor
from torch.utils.data.dataset import Dataset
from typing_extensions import TypeVar

from torchwrench.extras.pandas import pd
from torchwrench.extras.speechbrain import DynamicItemDataset
from torchwrench.nn.functional.multilabel import multihot_to_multi_indices
from torchwrench.types import IntegralTensor0D

T_RowIndex = TypeVar("T_RowIndex", bound=int, covariant=False, default=int)
T_ColIndex = TypeVar("T_ColIndex", bound=Union[int, str], covariant=False, default=str)


class SizedGenerator:
    def __init__(self, generator: Generator, size: int) -> None:
        super().__init__()
        self._generator = generator
        self._size = size

    def __iter__(self):
        yield from self._generator

    def __len__(self) -> int:
        return self._size


class TabularDatasetInterface(Dataset, Generic[T_RowIndex, T_ColIndex]):
    def __init__(self) -> None:
        super().__init__()

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

    def keys(self) -> pw.SupportsGetitemIterLen[T_ColIndex]:
        return self.column_names

    def values(self) -> SizedGenerator:
        return SizedGenerator((self[k] for k in self.keys()), self.num_columns)

    def concat_columns_with(
        self,
        other: "TabularDatasetInterface[T_RowIndex, T_ColIndex]",
    ) -> "ColumnConcatWrapper[T_RowIndex, T_ColIndex]":
        return ColumnConcatWrapper([self, other])

    def __iter__(self):
        for row_index in self.row_names:
            yield self[row_index]

    def __len__(self) -> int:
        return self.num_rows

    @property
    @abstractmethod
    def row_names(self) -> SupportsGetitemIterLen[T_RowIndex]:
        raise NotImplementedError

    @property
    @abstractmethod
    def column_names(self) -> SupportsGetitemIterLen[T_ColIndex]:
        raise NotImplementedError

    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def to_dict_list(self) -> Dict[T_ColIndex, List]:
        raise NotImplementedError

    @abstractmethod
    def to_list_dict(self) -> List[Dict[T_ColIndex, Any]]:
        raise NotImplementedError

    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def to_tensor(self) -> Tensor:
        raise NotImplementedError

    @overload
    def __getitem__(self, indexer: T_RowIndex, /) -> Dict[T_ColIndex, Any]: ...

    @overload
    def __getitem__(
        self,
        indexer: Union[
            Iterable[T_RowIndex], Iterable[T_ColIndex], Iterable[bool], slice
        ],
        /,
    ) -> List[Dict[T_ColIndex, Any]]: ...

    @overload
    def __getitem__(
        self, indexer: T_ColIndex, /
    ) -> Union[List[Any], np.ndarray, Tensor]: ...

    @overload
    def __getitem__(self, indexer: Tuple[T_RowIndex, T_ColIndex], /) -> Any: ...

    @abstractmethod
    def __getitem__(self, indexer) -> Any:
        raise NotImplementedError


class DictListWrapper(Generic[T_ColIndex], TabularDatasetInterface[int, T_ColIndex]):
    def __init__(self, data: Mapping[T_ColIndex, pw.SupportsGetitemIterLen]) -> None:
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
        return pd.DataFrame(self._data)  # type: ignore

    def to_dict_list(self) -> Dict[T_ColIndex, List]:
        return {k: list(v) for k, v in self._data.items()}

    def to_list_dict(self) -> List[Dict[T_ColIndex, Any]]:
        return pw.dict_list_to_list_dict(self._data, "same")

    def to_numpy(self) -> np.ndarray:
        return np.array(list(self._data.values())).T

    def to_tensor(self) -> Tensor:
        return torch.as_tensor(list(self._data.values())).T

    def __getitem__(self, indexer_, /):  # type: ignore
        indexer = IndexerWrapper(indexer_, self)
        del indexer_

        if indexer.single_col:
            result = self._data[indexer.col]  # type: ignore
            result = _get_from_idx_indices_slice_mask(result, indexer.row)
            return result

        result_dict = {
            col: _get_from_idx_indices_slice_mask(self._data[col], indexer.row)
            for col in indexer.col  # type: ignore
        }

        if indexer.has_col_indexer:
            return list(result_dict.values())
        elif indexer.single_row:
            return result_dict
        else:
            result_list = pw.dict_list_to_list_dict(result_dict, "same")
            return result_list


class DataFrameWrapper(TabularDatasetInterface[int, str]):
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

    def to_tensor(self) -> Tensor:
        return torch.from_numpy(self._data.to_numpy())

    def __getitem__(self, indexer, /):
        return self._data[indexer]


class TensorOrArrayWrapper(TabularDatasetInterface[int, int]):
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
        del id_column_name
        data_ids = list(range(len(self._data)))
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

    def to_tensor(self) -> Tensor:
        data = self._data
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return data

    def __getitem__(self, indexer, /):  # type: ignore
        return self._data[indexer]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape


class DynamicDatasetWrapper(TabularDatasetInterface[int, str]):
    def __init__(self, data: DynamicItemDataset) -> None:
        super().__init__()
        self._data = data

    @property
    def row_names(self) -> range:
        return range(len(self._data))

    @property
    def column_names(self) -> Tuple[str, ...]:
        return tuple(self._data.pipeline.output_mapping.keys())

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

    def to_tensor(self) -> Tensor:
        datalist = self[:]
        return torch.as_tensor([list(item.values()) for item in datalist])  # type: ignore

    def __getitem__(self, indexer_, /):  # type: ignore
        indexer = IndexerWrapper(indexer_, self)
        del indexer_

        output_keys = [indexer.col] if indexer.single_col else indexer.col
        with self._data.output_keys_as(output_keys):
            result = _get_from_idx_indices_slice_mask(self._data, indexer.row)

        if indexer.single_col and indexer.single_row:
            result = result[indexer.col]
            return result
        elif indexer.single_col and not indexer.single_row:
            result = [result_i[indexer.col] for result_i in result]
            return result
        elif indexer.has_col_indexer and indexer.single_row:
            result = list(result.values())
            return result
        elif indexer.has_col_indexer and not indexer.single_row:
            result = [list(result_i.values()) for result_i in result]
            return result
        elif not indexer.single_col:
            return result
        else:
            msg = f"Invalid state {indexer.single_row=}, {indexer.single_col=}, {indexer.has_col_indexer=}."
            raise TypeError(msg)


class FunctionWrapper(
    Generic[T_RowIndex, T_ColIndex],
    TabularDatasetInterface[T_RowIndex, T_ColIndex],
):
    def __init__(
        self,
        ds: TabularDatasetInterface[T_RowIndex, T_ColIndex],
        fns_list: Iterable[
            Tuple[
                Union[Tuple[T_ColIndex, ...], T_ColIndex],
                Union[Tuple[T_ColIndex, ...], T_ColIndex],
                Callable,
            ]
        ],
    ) -> None:
        fns = _get_fns_dict(fns_list)

        super().__init__()
        self._ds = ds
        self._fns = fns

    def add_dynamic_column(
        self,
        fn: Callable,
        requires: Tuple[T_ColIndex, ...],
        provides: Union[T_ColIndex, Tuple[T_ColIndex, ...]],
    ) -> None:
        provides_list = [provides] if isinstance(provides, (int, str)) else provides
        invalid = [provide for provide in provides_list if provide in self._fns]
        if len(invalid) > 0:
            msg = f"Found values already provided. (with {invalid=})"
            raise ValueError(msg)

        for provide in provides_list:
            self._fns[provide] = (requires, provides, fn)

    @property
    def row_names(self) -> range:
        return range(len(self._ds))

    @property
    def column_names(self) -> Tuple[T_ColIndex, ...]:
        return tuple(self._ds.column_names) + tuple(self._fns.keys())

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_dict_list())

    def to_dict_list(self) -> Dict[T_ColIndex, List]:
        datalist = self.to_list_dict()
        return pw.list_dict_to_dict_list(datalist)

    def to_list_dict(self) -> List[Dict[T_ColIndex, Any]]:
        datalist = self[:]
        return datalist  # type: ignore

    def to_numpy(self) -> np.ndarray:
        datalist = self.to_list_dict()
        return np.array([list(item.values()) for item in datalist])

    def to_tensor(self) -> Tensor:
        datalist = self.to_list_dict()
        return torch.as_tensor([list(item.values()) for item in datalist])

    def __getitem__(self, indexer_, /) -> Any:
        indexer = IndexerWrapper(indexer_, self)
        del indexer_

        row_indexer = indexer.row
        col_indexer: Iterable[T_ColIndex] = (
            [indexer.col] if indexer.single_col else indexer.col
        )  # type: ignore

        if isinstance(row_indexer, int):
            result = self._get_values(row_indexer, col_indexer)  # type: ignore
        elif isinstance(row_indexer, slice):
            row_indexer = range(len(self))[row_indexer]
            result = [self._get_values(idx, col_indexer) for idx in row_indexer]  # type: ignore
        elif pw.isinstance_generic(row_indexer, Iterable[bool]):
            row_indexer = multihot_to_multi_indices(row_indexer)
            result = [self._get_values(idx, col_indexer) for idx in row_indexer]  # type: ignore
        elif pw.isinstance_generic(row_indexer, Iterable[int]):
            result = [self._get_values(idx, col_indexer) for idx in row_indexer]  # type: ignore
        else:
            msg = f"Invalid argument type {type(row_indexer)=}."
            raise TypeError(msg)

        if not indexer.has_col_indexer:
            return result
        elif indexer.single_row and indexer.single_col:
            return result[indexer.col]  # type: ignore
        elif indexer.single_row:
            return [result[col] for col in indexer.col]  # type: ignore
        elif indexer.single_col:
            return [result_i[indexer.col] for result_i in result]  # type: ignore
        else:
            return [[result_i[col] for col in indexer.col] for result_i in result]  # type: ignore

    def _get_values(
        self,
        idx: T_RowIndex,
        columns: Iterable[T_ColIndex],
    ) -> Dict[T_ColIndex, Any]:
        return _recursive_get_values(idx, columns, self._ds, self._fns)


def _get_fns_dict(
    fns_list: Iterable[
        Tuple[
            Union[Tuple[T_ColIndex, ...], T_ColIndex],
            Union[Tuple[T_ColIndex, ...], T_ColIndex],
            Callable,
        ]
    ],
) -> Dict[
    T_ColIndex,
    Tuple[
        Union[Tuple[T_ColIndex, ...], T_ColIndex],
        Union[Tuple[T_ColIndex, ...], T_ColIndex],
        Callable,
    ],
]:
    fns = {}
    for requires, provides, fn in fns_list:
        if isinstance(provides, (int, str)):
            fns[provides] = (requires, provides, fn)  # type: ignore
            continue

        for provide in provides:
            fns[provide] = (requires, provides, fn)

    return fns


def _recursive_get_values(
    idx: T_RowIndex,
    columns: Iterable[T_ColIndex],
    ds: TabularDatasetInterface[T_RowIndex, T_ColIndex],
    fns: Dict[T_ColIndex, Tuple],
) -> Dict[T_ColIndex, Any]:
    result = {}
    for col in columns:
        if col in result:
            continue

        if col in ds.column_names:
            result[col] = ds[idx, col]
            continue

        if col not in fns.keys():
            raise KeyError

        requires, provides, fn = fns[col]
        if isinstance(requires, (int, str)):
            requires = [requires]

        required_values_dict = _recursive_get_values(idx, requires, ds, fns)  # type: ignore
        required_values_list = [required_values_dict[k] for k in requires]
        provided_values_list = fn(*required_values_list)

        if isinstance(provides, (int, str)):
            provided_values_dict = {provides: provided_values_list}
        else:
            provided_values_dict = dict(zip(provides, provided_values_list))

        result.update(required_values_dict)
        result.update(provided_values_dict)

    return result


class ColumnConcatWrapper(
    Generic[T_RowIndex, T_ColIndex],
    TabularDatasetInterface[T_RowIndex, T_ColIndex],
):
    def __init__(
        self,
        tabulars: Iterable[TabularDatasetInterface[T_RowIndex, T_ColIndex]],
    ) -> None:
        tabulars = list(tabulars)

        super().__init__()
        self._dss = tabulars

    @property
    def row_names(self) -> SupportsGetitemIterLen:
        if len(self._dss) == 0:
            return []
        return self._dss[0].row_names

    @property
    def column_names(self) -> SupportsGetitemIterLen:
        return [col for tabular in self._dss for col in tabular.column_names]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.concat(
            [tabular.to_dataframe() for tabular in self._dss], axis="columns"
        )

    def to_dict_list(self) -> Dict[Any, List]:
        return pw.reduce_or([tabular.to_dict_list() for tabular in self._dss], start={})

    def to_list_dict(self) -> List[Dict]:
        return pw.dict_list_to_list_dict(self.to_dict_list())

    def to_numpy(self) -> np.ndarray:
        return np.concat([tabular.to_numpy() for tabular in self._dss], axis=0)

    def to_tensor(self) -> Tensor:
        return torch.cat([tabular.to_tensor() for tabular in self._dss], dim=0)

    def __getitem__(self, indexer_, /) -> Any:
        indexer = IndexerWrapper(indexer_, self)
        del indexer_

        if indexer.single_col:
            for ds in self._dss:
                if indexer.col in ds.column_names:  # type: ignore
                    return ds[indexer.row, indexer.col]  # type: ignore
            raise TypeError

        result = []
        for ds in self._dss:
            ds_cols = [col for col in indexer.col if col in ds.column_names]  # type: ignore
            if len(ds_cols) == 0:
                continue

            result_i = ds[indexer.row, ds_cols]  # type: ignore
            result.append(result_i)

        if indexer.single_row:
            result = pw.reduce_or(result, start={})
            if indexer.has_col_indexer:
                return list(result.values())
            else:
                return result

        else:
            result_size = len(result[0]) if len(result) > 0 else 0
            result = [
                pw.reduce_or((result_i[j] for result_i in result), start={})
                for j in range(result_size)
            ]
            if indexer.has_col_indexer:
                return [list(result_i.values()) for result_i in result]
            else:
                return result


class IndexerWrapper:
    def __init__(
        self,
        indexer: Any,
        dataset: TabularDatasetInterface[Any, Any],
    ) -> None:
        if isinstance(indexer, IndexerWrapper):
            row_indexer = indexer._row_indexer
            col_indexer = indexer._col_indexer
            has_col_indexer = indexer._has_col_indexer

        else:
            if pw.isinstance_generic(
                indexer, (int, slice, Iterable[int], Iterable[bool])
            ):
                row_indexer = indexer
                col_indexer = None
            elif isinstance(indexer, tuple) and len(indexer) == 2:
                row_indexer, col_indexer = indexer
            else:
                row_indexer = slice(None)
                col_indexer = indexer

            has_col_indexer = col_indexer is not None

        if col_indexer is None:
            col_indexer = dataset.column_names

        super().__init__()
        self._row_indexer = row_indexer
        self._col_indexer = col_indexer
        self._has_col_indexer = has_col_indexer

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._row_indexer}, {self._col_indexer}, {self._has_col_indexer})"

    @property
    def row(self) -> Union[int, slice, Iterable[int], Iterable[bool]]:
        return self._row_indexer

    @property
    def col(self) -> Union[int, str, Iterable[int], Iterable[str]]:
        return self._col_indexer

    @property
    def single_row(self) -> bool:
        return isinstance(self._row_indexer, (int,))

    @property
    def single_col(self) -> bool:
        return isinstance(self._col_indexer, (int, str))

    @property
    def has_col_indexer(self) -> bool:
        return self._has_col_indexer


def _get_from_idx_indices_slice_mask(
    x: Union[Tensor, np.ndarray, pw.SupportsGetitemLen],
    row_indexer: Union[int, Tensor, np.ndarray, slice, Iterable[int], Iterable[bool]],
) -> Any:
    if isinstance(x, DynamicItemDataset) and isinstance(row_indexer, slice):
        indices = range(len(x))[row_indexer]
        return [x[idx] for idx in indices]
    elif isinstance(row_indexer, (int, slice)):
        return x[row_indexer]  # type: ignore
    elif isinstance(row_indexer, (IntegralTensor0D, np.integer)):
        return x[row_indexer.item()]  # type: ignore
    elif pw.isinstance_generic(row_indexer, Iterable[int]):
        return _get_from_indices(x, row_indexer)
    elif pw.isinstance_generic(row_indexer, Iterable[bool]):
        return _get_from_mask(x, row_indexer)
    else:
        msg = f"Invalid argument type {type(row_indexer)=}."
        raise TypeError(msg)


def _get_from_indices(
    x: Union[Tensor, np.ndarray, pw.SupportsGetitemLen],
    indices: Union[Tensor, np.ndarray, Iterable],
) -> Union[Tensor, np.ndarray, list]:
    if isinstance(x, Tensor) and isinstance(indices, Tensor):
        return x[indices]
    elif isinstance(x, np.ndarray) and isinstance(indices, np.ndarray):
        return x[indices]
    elif isinstance(x, pw.SupportsGetitemLen):
        return [x[idx] for idx in indices]  # type: ignore
    else:
        msg = f"Invalid argument type {type(indices)=}."
        raise TypeError(msg)


def _get_from_mask(
    x: Union[Tensor, np.ndarray, pw.SupportsGetitemLen],
    mask: Union[Tensor, np.ndarray, Iterable],
) -> Union[Tensor, np.ndarray, list]:
    if isinstance(x, Tensor) and isinstance(mask, Tensor):
        return x[mask]
    elif isinstance(x, np.ndarray) and isinstance(mask, np.ndarray):
        return x[mask]
    elif isinstance(x, pw.SupportsGetitemLen):
        indices = multihot_to_multi_indices(mask)
        return [x[idx] for idx in indices]
    else:
        msg = f"Invalid argument type {type(mask)=}."
        raise TypeError(msg)
