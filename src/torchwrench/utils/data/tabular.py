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
from typing_extensions import Self, TypeVar

import torchwrench as tw
from torchwrench.extras.numpy import np

T = TypeVar("T", covariant=True, default=Any)
T_Item = TypeVar("T_Item", covariant=True, default=Any)
T_Metadata = TypeVar("T_Metadata", covariant=False, default=Any)
T_Index = TypeVar("T_Index", covariant=True, default=int)
T_Column = TypeVar("T_Column", covariant=False, default=str)


class TabularDataset(Generic[T_Item, T_Index, T_Column, T_Metadata], Dataset[T_Item]):
    def __init__(
        self,
        data,
        output_keys: Iterable[T_Column],
        dynamic_fns: Iterable[
            Tuple[Tuple[T_Column, ...], Tuple[T_Column, ...], Callable]
        ],
        metadata: T_Metadata,
    ) -> None:
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
        output_keys: Optional[Iterable[T_Column]] = None,
        metadata: T_Metadata = None,
    ) -> "TabularDataset[Dict[T_Column, T], int, T_Column, T_Metadata]":
        if output_keys is None:
            output_keys = list(df.keys())

        return TabularDataset(
            df,
            output_keys=output_keys,
            dynamic_fns={},
            metadata=metadata,
        )

    @classmethod
    def from_dict(
        cls,
        dic: Mapping[T_Column, SupportsGetitemLen[T]],
        *,
        output_keys: Optional[Iterable[T_Column]] = None,
        metadata: T_Metadata = None,
    ) -> "TabularDataset[Dict[T_Column, T], int, T_Column, T_Metadata]":
        if output_keys is None:
            output_keys = list(dic.keys())

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
        output_keys: Optional[Iterable[T_Column]] = None,
        metadata: T_Metadata = None,
    ) -> "TabularDataset[Dict[T_Column, T], int, T_Column, T_Metadata]":
        if output_keys is None:
            output_keys = list(dset.pipeline.output_mapping.keys())

        return TabularDataset(
            dset,
            output_keys=output_keys,
            dynamic_fns={},
            metadata=metadata,
        )

    @classmethod
    def from_list(
        cls,
        lst: List[Dict[T_Column, Any]],
        *,
        output_keys: Optional[Iterable[T_Column]] = None,
        metadata: T_Metadata = None,
    ) -> "TabularDataset[Dict[T_Column, Any], int, T_Column, T_Metadata]":
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
    def column_names(self) -> Tuple[T_Column, ...]:
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
    def static_keys(self) -> Tuple[T_Column, ...]:
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
    def dynamic_keys(self) -> Tuple[T_Column, ...]:
        return tuple(key for _, provides, _ in self._dynamic_fns for key in provides)

    @property
    def output_keys(self) -> Tuple[T_Column, ...]:
        return tuple(self._output_keys)

    def keys(self) -> Tuple[T_Column, ...]:
        return tuple(self._output_keys)

    def values(self):
        raise NotImplementedError

    def items(self):
        raise NotImplementedError

    def add_dynamic_column(self, fn, takes, provides, batch) -> None:
        self._dynamic_fns.append((takes, provides, fn))

    def add_column(self, key: T_Column, column_data: Any) -> None:
        if isinstance(self._data, (pd.DataFrame, dict)):
            self._data[key] = column_data
        elif isinstance(self._data, list):
            for data_i, column_data_i in zip(self._data, column_data):
                data_i[key] = column_data_i
        elif isinstance(self._data, DynamicItemDataset):
            for sample, column_data_i in zip(self._data.data.values(), column_data):
                sample[key] = column_data_i
        else:
            raise TypeError

    def remove_column(self, key: T_Column):
        raise NotImplementedError

    def rename_column(self, old_key: T_Column, new_key: T_Column):
        raise NotImplementedError

    def set_output_keys(self, keys: Iterable[T_Column]) -> None:
        keys = list(keys)
        self._output_keys = keys

    def add_output_keys(self, keys: Iterable[T_Column]) -> None:
        keys = list(keys)
        self._output_keys += keys

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

    def to_dict(self) -> Dict[T_Column, List[Any]]:
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
        id_column_key: Optional[T_Column] = None,
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
            T_Column,
            Iterable[T_Column],
        ],
    ) -> Any:
        raise NotImplementedError

    def __setitem__(self, index, obj) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.num_rows

    def __repr__(self) -> str:
        repr_ = (
            f"{self.__class__.__name__}(num_rows={self.num_rows}, keys={self.keys()})"
        )
        return repr_
