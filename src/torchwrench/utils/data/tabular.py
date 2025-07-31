#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import pandas as pd
import pythonwrench as pw
from pythonwrench.typing.classes import SupportsGetitemLen
from speechbrain.dataio.dataset import DynamicItemDataset
from torch import Tensor
from torch.utils.data.dataset import Dataset
from typing_extensions import TypeIs, TypeVar

import torchwrench as tw
from torchwrench.extras.numpy import is_numpy_bool_array, is_numpy_integral_array, np

T = TypeVar("T", covariant=True, default=Any)
T_Item = TypeVar("T_Item", covariant=True, default=Any)
T_Metadata = TypeVar("T_Metadata", covariant=False, default=Any)
T_Index = TypeVar("T_Index", covariant=True, default=int)
T_ColumnKey = TypeVar("T_ColumnKey", covariant=False, default=str)


class TabularDataset(
    Generic[T_Item, T_Index, T_ColumnKey, T_Metadata], Dataset[T_Item]
):
    def __init__(
        self,
        data=None,
        output_keys: Optional[Iterable[T_ColumnKey]] = None,
        dynamic_fns: Iterable[
            Tuple[Tuple[T_ColumnKey, ...], Tuple[T_ColumnKey, ...], Callable]
        ] = (),
        metadata: T_Metadata = None,
    ) -> None:
        if data is None:
            data = {}
        if output_keys is None:
            output_keys = _get_static_keys(data)
        output_keys = list(output_keys)
        dynamic_fns = list(dynamic_fns)

        super().__init__()
        self._data = data
        self._output_keys = output_keys
        self._dynamic_fns = dynamic_fns
        self._metadata = metadata

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        output_keys: Optional[Iterable[str]] = None,
        metadata: T_Metadata = None,
    ) -> "TabularDataset[Dict[str, T], int, str, T_Metadata]":
        return TabularDataset(
            df,
            output_keys=output_keys,
            dynamic_fns={},
            metadata=metadata,
        )

    @classmethod
    def from_dict(
        cls,
        dic: Mapping[T_ColumnKey, SupportsGetitemLen[T]],
        *,
        output_keys: Optional[Iterable[T_ColumnKey]] = None,
        metadata: T_Metadata = None,
    ) -> "TabularDataset[Dict[T_ColumnKey, T], int, T_ColumnKey, T_Metadata]":
        return TabularDataset(
            dic,
            output_keys=output_keys,
            dynamic_fns={},
            metadata=metadata,
        )

    @classmethod
    def from_dynamic_item_dataset(
        cls,
        dset: DynamicItemDataset,
        *,
        output_keys: Optional[Iterable[T_ColumnKey]] = None,
        metadata: T_Metadata = None,
    ) -> "TabularDataset[Dict[T_ColumnKey, T], int, T_ColumnKey, T_Metadata]":
        return TabularDataset(
            dset,
            output_keys=output_keys,
            dynamic_fns={},
            metadata=metadata,
        )

    @classmethod
    def from_list(
        cls,
        lst: List[Dict[T_ColumnKey, Any]],
        *,
        output_keys: Optional[Iterable[T_ColumnKey]] = None,
        metadata: T_Metadata = None,
    ) -> "TabularDataset[Dict[T_ColumnKey, Any], int, T_ColumnKey, T_Metadata]":
        if output_keys is None:
            if len(lst) == 0:
                output_keys = []
            else:
                output_keys = list(lst[0].keys())

        return TabularDataset(
            lst,
            output_keys=output_keys,
            dynamic_fns={},
            metadata=metadata,
        )

    @classmethod
    def from_matrix(
        cls,
        matrix: Union[List[List[Any]], np.ndarray, Tensor],
        *,
        output_keys: Optional[Iterable[int]] = None,
        metadata: T_Metadata = None,
    ) -> "TabularDataset[Dict[int, Any], int, int, T_Metadata]":
        if output_keys is None:
            output_keys = list(range(len(matrix)))

        return TabularDataset(
            matrix,
            output_keys=output_keys,
            dynamic_fns={},
            metadata=metadata,
        )

    @property
    def column_names(self) -> Tuple[T_ColumnKey, ...]:
        return tuple(self._output_keys)

    @property
    def metadata(self) -> T_Metadata:
        return self._metadata

    @property
    def num_columns(self) -> int:
        return len(self.column_names)

    @property
    def num_rows(self) -> int:
        if isinstance(
            self._data, (pd.DataFrame, DynamicItemDataset, list, Tensor, np.ndarray)
        ):
            return len(self._data)
        elif isinstance(self._data, dict):
            if len(self._data) == 0:
                return 0
            else:
                return len(next(iter(self._data.values())))
        else:
            raise TypeError

    @property
    def shape(self) -> Tuple[int, int]:
        return self.num_rows, self.num_columns

    @property
    def static_keys(self) -> Tuple[T_ColumnKey, ...]:
        if isinstance(self._data, (pd.DataFrame, dict)):
            return tuple(self._data.keys())
        elif isinstance(self._data, DynamicItemDataset):
            if len(self._data.data) == 0:
                return ()
            else:
                return tuple(next(iter(self._data.data.values())))
        elif isinstance(self._data, (Tensor, np.ndarray)):
            return tuple(range(len(self._data)))  # type: ignore
        elif isinstance(self._data, list):
            if len(self._data) == 0:
                return ()
            else:
                return tuple(self._data[0].keys())
        else:
            raise TypeError

    @property
    def dynamic_keys(self) -> Tuple[T_ColumnKey, ...]:
        return tuple(key for _, provides, _ in self._dynamic_fns for key in provides)

    @property
    def output_keys(self) -> Tuple[T_ColumnKey, ...]:
        return tuple(self._output_keys)

    def keys(self) -> Tuple[T_ColumnKey, ...]:
        return tuple(self._output_keys)

    def values(self) -> Iterable:
        if isinstance(self._data, (dict, pd.DataFrame)):
            return [self._data[k] for k in self.keys()]
        elif pw.isinstance_generic(self._data, List[Dict]):
            return [[data_i[k] for data_i in self._data] for k in self.keys()]
        elif isinstance(self._data, DynamicItemDataset):
            return [self._data[i] for i in range(len(self._data))]
        else:
            raise TypeError

    def items(self) -> Iterable[Tuple[T_ColumnKey, Any]]:
        return zip(self.keys(), self.values())

    def add_dynamic_column(self, fn, takes, provides, batch) -> None:
        self._dynamic_fns.append((takes, provides, fn))

    def add_column(
        self, key: T_ColumnKey, column_data: Any, add_to_output_keys: bool = True
    ) -> None:
        if isinstance(self._data, (pd.DataFrame, dict)):
            self._data[key] = column_data
        elif pw.isinstance_generic(self._data, List[Dict]):
            for data_i, column_data_i in zip(self._data, column_data):
                data_i[key] = column_data_i
        elif isinstance(self._data, DynamicItemDataset):
            for sample, column_data_i in zip(self._data.data.values(), column_data):
                sample[key] = column_data_i
        else:
            raise TypeError

        if add_to_output_keys:
            self.add_output_keys([key])

    def pop_column(self, key: T_ColumnKey) -> Any:
        if isinstance(self._data, pd.DataFrame):
            column_data = self._data.pop(key)  # type: ignore
        if isinstance(self._data, dict):
            column_data = self._data.pop(key)
        elif pw.isinstance_generic(self._data, List[Dict]):
            column_data = []
            for data_i in self._data:
                column_data.append(data_i.pop(key))
        elif isinstance(self._data, DynamicItemDataset):
            column_data = []
            for data_i in self._data.data.values():
                column_data.append(data_i.pop(key))
        else:
            raise TypeError

        return column_data

    def rename_column(
        self,
        old_key: T_ColumnKey,
        new_key: T_ColumnKey,
        add_to_output_keys: bool = True,
    ) -> None:
        column_data = self.pop_column(old_key)
        self.add_column(new_key, column_data, add_to_output_keys=add_to_output_keys)

    def set_output_keys(self, keys: Iterable[T_ColumnKey]) -> None:
        keys = list(keys)
        self._output_keys = keys

    def add_output_keys(self, keys: Iterable[T_ColumnKey]) -> None:
        keys = list(keys)
        self._output_keys += keys

    def pop_output_keys(self, keys: Iterable[T_ColumnKey]) -> None:
        for key in keys:
            self._output_keys.remove(key)

    def to_dataframe(self) -> pd.DataFrame:
        if isinstance(self._data, pd.DataFrame):
            return self._data
        elif isinstance(self._data, (list, dict)):
            return pd.DataFrame(self._data)
        elif isinstance(self._data, DynamicItemDataset):
            ids = self._data.data.keys()
            lst = self._data.data.values()
            return pd.DataFrame(lst, index=ids)
        else:
            raise TypeError

    def to_dict(self) -> Dict[T_ColumnKey, List[Any]]:
        if isinstance(self._data, pd.DataFrame):
            return self._data.to_dict("list")  # type: ignore
        elif isinstance(self._data, list):
            return pw.list_dict_to_dict_list(self._data, "same")
        elif isinstance(self._data, dict):
            return self._data
        elif isinstance(self._data, DynamicItemDataset):
            return pw.list_dict_to_dict_list(self._data.data.values(), "same")
        else:
            raise TypeError

    def to_list(self) -> List[T_Item]:
        if isinstance(self._data, pd.DataFrame):
            return self._data.to_dict("records")  # type: ignore
        elif isinstance(self._data, list):
            return self._data
        elif isinstance(self._data, dict):
            return pw.dict_list_to_list_dict(self._data, "same")  # type: ignore
        elif isinstance(self._data, DynamicItemDataset):
            return list(self._data.data.values())
        else:
            raise TypeError

    def to_matrix(self) -> np.ndarray:
        if isinstance(self._data, pd.DataFrame):
            return self._data.to_numpy()
        elif isinstance(self._data, list):
            return np.array([list(data_i.values() for data_i in self._data)])
        elif isinstance(self._data, dict):
            return np.array([list(data_i) for data_i in self._data.values()])
        elif isinstance(self._data, DynamicItemDataset):
            return np.array(
                [list(data_i.values()) for data_i in self._data.data.values()]
            )
        else:
            raise TypeError

    def to_dynamic_item_dataset(
        self,
        id_column_key: Optional[T_ColumnKey] = None,
    ) -> DynamicItemDataset:
        if isinstance(self._data, pd.DataFrame):
            if id_column_key is not None and id_column_key in self._data:
                ids = self._data[id_column_key]
            else:
                ids = list(range(len(self._data)))

            did_data = {
                id_: data_i.to_dict()
                for id_, (_, data_i) in zip(ids, self._data.iterrows())
            }
            return DynamicItemDataset(did_data)

        elif isinstance(self._data, list):
            if len(self._data) == 0:
                return DynamicItemDataset({})

            keys = list(self._data[0].keys())
            if id_column_key is not None and id_column_key in keys:
                ids = [data_i[id_column_key] for data_i in self._data]
            else:
                ids = list(range(len(self._data)))

            did_data = {id_: data_i for id_, data_i in zip(ids, self._data)}
            return DynamicItemDataset(did_data)

        elif isinstance(self._data, dict):
            if id_column_key is not None and id_column_key in self._data.keys():
                ids = self._data[id_column_key]
            else:
                ids = list(range(len(self._data)))

            did_data = {
                id_: {k: v[i] for k, v in self._data.items()}
                for i, id_ in enumerate(ids)
            }
            return DynamicItemDataset(did_data)

        elif isinstance(self._data, DynamicItemDataset):
            return self._data

        else:
            raise TypeError

    def __getitem__(
        self,
        index: Union[
            T_Index,
            Iterable[T_Index],
            Iterable[bool],
            Tensor,
            np.ndarray,
            slice,
            T_ColumnKey,
            Iterable[T_ColumnKey],
        ],
    ) -> Any:
        if isinstance(self._data, pd.DataFrame):
            if tw.is_number_like(index) or is_mask(index) or isinstance(index, slice):
                index = tw.to_ndarray(index)
                return self._data[index]
            elif is_indices(index):
                mask = tw.multi_indices_to_multihot(index, len(self)).numpy()
                return self[mask]
            elif isinstance(index, str):
                return self._data[index]
            elif pw.isinstance_generic(index, Iterable[str]):
                return self._data[list(index)]
            else:
                raise TypeError

        elif pw.isinstance_generic(self._data, List[Dict]):
            if tw.is_number_like(index) or isinstance(index, slice):
                index = tw.as_builtin(index)  # type: ignore
                return self._data[index]  # type: ignore
            elif is_indices(index):
                index = tw.as_builtin(index)
                return [self[index_i] for index_i in index]  # type: ignore
            elif is_mask(index):
                return [self[i] for i, mask_i in enumerate(index) if mask_i]  # type: ignore
            elif isinstance(index, str):
                return [sample[index] for sample in self._data]
            elif pw.isinstance_generic(index, Iterable[str]):
                return [
                    {index_i: sample[index_i] for index_i in index}
                    for sample in self._data
                ]
            else:
                raise TypeError

        elif pw.isinstance_generic(self._data, Dict[str, list]):
            if tw.is_number_like(index) or isinstance(index, slice):
                index = tw.as_builtin(index)  # type: ignore
                return {k: v[index] for k, v in self._data.items()}  # type: ignore
            elif is_indices(index):
                index = tw.as_builtin(index)
                return {
                    k: [v[index_i] for index_i in index] for k, v in self._data.items()
                }  # type: ignore
            elif is_mask(index):
                return {
                    k: [v[i] for i, mask_i in enumerate(index) if mask_i]
                    for k, v in self._data.items()
                }  # type: ignore
            elif isinstance(index, str):
                return self._data[index]
            elif pw.isinstance_generic(index, Iterable[str]):
                return {index_i: self._data[index_i] for index_i in index}
            else:
                raise TypeError

        elif pw.isinstance_generic(self._data, DynamicItemDataset):
            if tw.is_number_like(index) or isinstance(index, slice):
                index = tw.as_builtin(index)  # type: ignore
                return self._data[index]  # type: ignore n
            elif is_indices(index):
                index = tw.as_builtin(index)
                return [self[index_i] for index_i in index]  # type: ignore
            elif is_mask(index):
                return [self[i] for i, mask_i in enumerate(index) if mask_i]  # type: ignore
            elif isinstance(index, str):
                return [sample[index] for sample in self._data]
            elif pw.isinstance_generic(index, Iterable[str]):
                return [
                    {index_i: sample[index_i] for index_i in index}
                    for sample in self._data
                ]
            else:
                raise TypeError

        else:
            raise TypeError

    def __len__(self) -> int:
        return self.num_rows

    def __repr__(self) -> str:
        repr_ = (
            f"{self.__class__.__name__}(num_rows={self.num_rows}, keys={self.keys()})"
        )
        return repr_


def _get_static_keys(
    data: Union[
        pd.DataFrame,
        List[Dict[T_ColumnKey, Any]],
        Dict[T_ColumnKey, List],
        DynamicItemDataset,
    ],
) -> Tuple[T_ColumnKey, ...]:
    if isinstance(data, pd.DataFrame):
        return tuple(data.keys())
    elif pw.isinstance_generic(data, List[Dict]):
        if len(data) == 0:
            return ()
        else:
            return tuple(data[0].keys())
    elif isinstance(data, dict):
        return tuple(data.keys())
    elif isinstance(data, DynamicItemDataset):
        return tuple(data.pipeline.output_mapping.keys())
    else:
        raise TypeError


def _get_dynamic_keys(
    data: Union[
        pd.DataFrame,
        List[Dict[T_ColumnKey, Any]],
        Dict[T_ColumnKey, List],
        DynamicItemDataset,
    ],
) -> Tuple[T_ColumnKey, ...]:
    if pw.isinstance_generic(data, (pd.DataFrame, List[Dict], dict)):
        return ()
    elif isinstance(data, DynamicItemDataset):
        return tuple(data.pipeline.output_mapping.keys())
    else:
        raise TypeError


def is_mask(x: Any) -> TypeIs[Union[Iterable[bool], np.ndarray, tw.BoolTensor1D]]:
    return (
        pw.isinstance_generic(x, Iterable[bool])
        or (is_numpy_bool_array(x) and x.ndim == 1)
        or isinstance(x, tw.BoolTensor1D)
    )


def is_indices(x: Any) -> TypeIs[Union[Iterable[int], np.ndarray, tw.IntegralTensor1D]]:
    return (
        pw.isinstance_generic(x, Iterable[int])
        or (is_numpy_integral_array(x) and x.ndim == 1)
        or isinstance(x, tw.IntegralTensor1D)
    )
