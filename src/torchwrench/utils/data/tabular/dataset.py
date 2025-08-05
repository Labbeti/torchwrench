#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
    overload,
)

import pythonwrench as pw
from pythonwrench.typing.classes import SupportsGetitemLen
from torch import Tensor
from torch.utils.data.dataset import Dataset
from typing_extensions import TypeVar

import torchwrench as tw
from torchwrench.extras.numpy import _NUMPY_AVAILABLE, np
from torchwrench.extras.pandas import pd
from torchwrench.extras.speechbrain import DynamicItemDataset
from torchwrench.utils.data.tabular._typechecks import (
    ColumnIndexer,
    MultiColumns,
    MultiIndexer,
    MultiRows,
    RowIndexer,
    SingleColumn,
    SingleIndexer,
    is_column_indexer,
    is_mask,
    is_multi_columns,
    is_multi_indices,
    is_multi_names,
    is_row_indexer,
    is_single_column,
    is_single_index,
    is_single_indexer,
)

T = TypeVar("T", covariant=True, default=Any)
U = TypeVar("U", covariant=True, default=Any)
V = TypeVar("V", covariant=True, default=Any)

T_Index = TypeVar("T_Index", bound=int, covariant=True, default=int)
T_ColumnKey = TypeVar("T_ColumnKey", int, str, covariant=False, default=str)
T_Value = TypeVar("T_Value", covariant=True, default=Any)
T_Item = TypeVar("T_Item", covariant=True, default=Any)
T_Metadata = TypeVar("T_Metadata", covariant=True, default=Any)

T_Index2 = TypeVar("T_Index2", bound=int, covariant=True, default=int)
T_ColumnKey2 = TypeVar("T_ColumnKey2", int, str, covariant=False, default=str)
T_Value2 = TypeVar("T_Value2", covariant=True, default=Any)
T_Item2 = TypeVar("T_Item2", covariant=True, default=Any)
T_Metadata2 = TypeVar("T_Metadata2", covariant=True, default=Any)


class TabularDataset(
    Generic[T_Index, T_ColumnKey, T_Value, T_Item, T_Metadata],
    Dataset[T_Item],
):
    """TabularDataset wrapper.

    This class wrap a dataset-like object and provide an unifsed interface to it.
    """

    @overload
    def __init__(
        self: "TabularDataset[int, str, Any, Dict[str, Any], T_Metadata2]",
        data: pd.DataFrame,
        *,
        output_columns: Optional[Iterable[str]] = None,
        dynamic_fns: Optional[
            Iterable[Tuple[Tuple[str, ...], Tuple[str, ...], Callable, bool]]
        ] = None,
        metadata: T_Metadata2 = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "TabularDataset[int, T_ColumnKey2, T_Value2, Dict[T_ColumnKey2, T_Value2], T_Metadata2]",
        data: Mapping[T_ColumnKey2, SupportsGetitemLen[T_Value2]],
        *,
        output_columns: Optional[Iterable[T_ColumnKey2]] = None,
        dynamic_fns: Optional[
            Iterable[
                Tuple[
                    Tuple[T_ColumnKey2, ...], Tuple[T_ColumnKey2, ...], Callable, bool
                ]
            ]
        ] = None,
        metadata: T_Metadata2 = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "TabularDataset[int, T_ColumnKey2, T_Value2, Dict[T_ColumnKey2, T_Value2], T_Metadata2]",
        data: List[Dict[T_ColumnKey2, T_Value2]],
        *,
        output_columns: Optional[Iterable[T_ColumnKey2]] = None,
        dynamic_fns: Optional[
            Iterable[
                Tuple[
                    Tuple[T_ColumnKey2, ...], Tuple[T_ColumnKey2, ...], Callable, bool
                ]
            ]
        ] = None,
        metadata: T_Metadata2 = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "TabularDataset[int, str, Any, Dict[str, Any], T_Metadata2]",
        data: DynamicItemDataset,
        *,
        output_columns: Optional[Iterable[str]] = None,
        dynamic_fns: Optional[
            Iterable[Tuple[Tuple[str, ...], Tuple[str, ...], Callable, bool]]
        ] = None,
        metadata: T_Metadata2 = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "TabularDataset[int, int, Union[np.ndarray, np.generic], np.ndarray, T_Metadata2]",
        data: np.ndarray,
        *,
        output_columns: Optional[Iterable[int]] = None,
        dynamic_fns: Optional[
            Iterable[Tuple[Tuple[int, ...], Tuple[int, ...], Callable, bool]]
        ] = None,
        metadata: T_Metadata2 = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "TabularDataset[int, int, Tensor, Tensor, T_Metadata2]",
        data: Tensor,
        *,
        output_columns: Optional[Iterable[int]] = None,
        dynamic_fns: Optional[
            Iterable[Tuple[Tuple[int, ...], Tuple[int, ...], Callable, bool]]
        ] = None,
        metadata: T_Metadata2 = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "TabularDataset[T_Index2, T_ColumnKey2, T_Value2, T_Item2, T_Metadata2]",
        data: "TabularDataset[T_Index2, T_ColumnKey2, T_Value2, T_Item2, T_Metadata2]",
        *,
        output_columns: Optional[Iterable[T_ColumnKey2]] = None,
        dynamic_fns: Optional[
            Iterable[
                Tuple[
                    Tuple[T_ColumnKey2, ...], Tuple[T_ColumnKey2, ...], Callable, bool
                ]
            ]
        ] = None,
        metadata: T_Metadata2 = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "TabularDataset[int, Any, Any, Dict[Any, Any], V]",
        data: Literal[None] = None,
        *,
        output_columns: Optional[Iterable[Any]] = None,
        dynamic_fns: Optional[
            Iterable[Tuple[Tuple[Any, ...], Tuple[Any, ...], Callable, bool]]
        ] = None,
        metadata: V = None,
    ) -> None: ...

    def __init__(
        self,
        data=None,
        *,
        output_columns: Optional[Iterable] = None,
        dynamic_fns: Optional[Iterable[Tuple[tuple, tuple, Callable, bool]]] = None,
        metadata: T_Metadata = None,
    ) -> None:
        if data is None:
            data = {}

        # Sanity check for input data
        if pw.isinstance_generic(data, Dict[Any, list]):
            uniq_sizes = set(map(len, data.values()))
            if len(uniq_sizes) not in (0, 1):
                msg = f"Invalid sizes in lists for data. (found different sizes {uniq_sizes=} but expected same size for all lists)"
                raise ValueError(msg)

        elif pw.isinstance_generic(data, List[Dict]):
            uniq_keys = set(set(data_i.keys()) for data_i in data)
            if len(uniq_keys) not in (0, 1):
                msg = f"Invalid keys in dicts for data. (found different keys {uniq_keys=} but expected same keys for all dicts)"
                raise ValueError(msg)

        elif isinstance(data, (np.ndarray, Tensor)):
            if data.ndim < 2:
                msg = f"Invalid number of dimensions for data. (found {data.ndim=} but expected >=2)"
                raise ValueError(msg)

        if output_columns is None:
            output_columns = _get_static_keys(data)

        if dynamic_fns is None:
            if isinstance(data, TabularDataset):
                dynamic_fns = data._dynamic_fns
            else:
                dynamic_fns = []

        output_columns = list(output_columns)
        dynamic_fns = list(dynamic_fns)

        super().__init__()
        self._data = data
        self._output_columns: List[T_ColumnKey] = output_columns  # type: ignore
        self._dynamic_fns = dynamic_fns
        self._metadata = metadata

    @property
    def column_names(self) -> Tuple[T_ColumnKey, ...]:
        return tuple(self._output_columns)

    @property
    def dynamic_keys(self) -> Tuple[T_ColumnKey, ...]:
        return tuple(key for _, provides, _, _ in self._dynamic_fns for key in provides)

    @property
    def metadata(self) -> T_Metadata:
        return self._metadata

    @property
    def num_columns(self) -> int:
        return len(self._output_columns)

    @property
    def num_rows(self) -> int:
        if pw.isinstance_generic(
            self._data,
            (pd.DataFrame, DynamicItemDataset, List[Dict], Tensor, np.ndarray),
        ):
            return len(self._data)

        elif pw.isinstance_generic(self._data, Dict[Any, list]):
            if len(self._data) == 0:
                return 0
            else:
                return len(next(iter(self._data.values())))

        else:
            msg = f"Invalid data type {type(self._data)=}."
            raise TypeError(msg)

    @property
    def output_columns(self) -> Tuple[T_ColumnKey, ...]:
        return tuple(self._output_columns)

    @property
    def output_keys(self) -> Tuple[T_ColumnKey, ...]:
        return tuple(self._output_columns)

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
            msg = f"Invalid data type {type(self._data)=}."
            raise TypeError(msg)

    def keys(self) -> Tuple[T_ColumnKey, ...]:
        return tuple(self._output_columns)

    def values(self) -> Iterable:
        if isinstance(self._data, (dict, pd.DataFrame)):
            return [self._data[k] for k in self.keys()]
        elif pw.isinstance_generic(self._data, List[Dict]):
            return [[data_i[k] for data_i in self._data] for k in self.keys()]
        elif isinstance(self._data, DynamicItemDataset):
            return [self._data[i] for i in range(len(self._data))]
        else:
            msg = f"Invalid data type {type(self._data)=}."
            raise TypeError(msg)

    def items(self) -> Iterable[Tuple[T_ColumnKey, Any]]:
        return zip(self.keys(), self.values())

    def add_dynamic_column(
        self,
        fn: Callable,
        takes: Tuple[T_ColumnKey, ...],
        provides: Tuple[T_ColumnKey, ...],
        batch: bool = False,
    ) -> None:
        self._dynamic_fns.append((takes, provides, fn, batch))

    def add_column(
        self,
        key: T_ColumnKey,
        column_data: Any,
        add_to_output_keys: bool = True,
    ) -> None:
        if pw.isinstance_generic(self._data, (pd.DataFrame, Dict[Any, List])):
            self._data[key] = column_data
        elif pw.isinstance_generic(self._data, List[Dict]):
            for data_i, column_data_i in zip(self._data, column_data):
                data_i[key] = column_data_i
        elif isinstance(self._data, DynamicItemDataset):
            for sample, column_data_i in zip(self._data.data.values(), column_data):
                sample[key] = column_data_i
        else:
            msg = f"Invalid data type {type(self._data)=}."
            raise TypeError(msg)

        if add_to_output_keys:
            self.add_output_keys([key])

    def pop_column(self, key: T_ColumnKey) -> Any:
        if isinstance(self._data, pd.DataFrame):
            column_data = self._data.pop(key)  # type: ignore
        if pw.isinstance_generic(self._data, Dict[Any, List]):
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
            msg = f"Invalid data type {type(self._data)=}."
            raise TypeError(msg)

        return column_data

    def rename_column(
        self,
        old_key: T_ColumnKey,
        new_key: T_ColumnKey,
        add_to_output_keys: bool = True,
    ) -> None:
        column_data = self.pop_column(old_key)
        self.add_column(new_key, column_data, add_to_output_keys=add_to_output_keys)

    def rename_columns(
        self,
        keys: Union[Mapping[T_ColumnKey, T_ColumnKey], Iterable[T_ColumnKey]],
    ) -> None:
        if isinstance(keys, Mapping):
            mapper = keys
        else:
            mapper = dict(zip(self.keys(), keys))

        for old_key, new_key in mapper.items():
            self.rename_column(old_key, new_key)

    def set_output_keys(self, keys: Iterable[T_ColumnKey]) -> None:
        keys = list(keys)
        self._output_columns = keys

    def add_output_keys(self, keys: Iterable[T_ColumnKey]) -> None:
        keys = list(keys)
        self._output_columns += keys

    def pop_output_keys(self, keys: Iterable[T_ColumnKey]) -> None:
        for key in keys:
            self._output_columns.remove(key)

    def to_dataframe(self) -> pd.DataFrame:
        if isinstance(self._data, pd.DataFrame):
            return self._data
        elif pw.isinstance_generic(
            self._data, (List[Dict], Dict[Any, pw.SupportsGetitemIterLen])
        ):
            return pd.DataFrame(self._data)
        elif isinstance(self._data, DynamicItemDataset):
            ids = self._data.data.keys()
            lst = self._data.data.values()
            return pd.DataFrame(lst, index=ids)
        else:
            msg = f"Invalid data type {type(self._data)=}."
            raise TypeError(msg)

    def to_dict(self) -> Dict[T_ColumnKey, pw.SupportsGetitemIterLen]:
        if isinstance(self._data, pd.DataFrame):
            return self._data.to_dict("list")  # type: ignore
        elif pw.isinstance_generic(self._data, List[Dict]):
            return pw.list_dict_to_dict_list(self._data, "same")  # type: ignore
        elif pw.isinstance_generic(self._data, Dict[Any, pw.SupportsGetitemIterLen]):
            return self._data
        elif isinstance(self._data, DynamicItemDataset):
            return pw.list_dict_to_dict_list(self._data.data.values(), "same")  # type: ignore
        elif isinstance(self._data, (np.ndarray, Tensor)):
            return dict(zip(self.keys(), map(list, self._data)))  # type: ignore
        else:
            msg = f"Invalid data type {type(self._data)=}."
            raise TypeError(msg)

    def to_list(self) -> List[T_Item]:
        if isinstance(self._data, pd.DataFrame):
            return self._data.to_dict("records")  # type: ignore
        elif pw.isinstance_generic(self._data, List[Dict]):
            return self._data  # type: ignore
        elif pw.isinstance_generic(self._data, Dict[Any, pw.SupportsGetitemIterLen]):
            return pw.dict_list_to_list_dict(self._data, "same")  # type: ignore
        elif isinstance(self._data, DynamicItemDataset):
            return list(self._data.data.values())
        elif isinstance(self._data, (np.ndarray, Tensor)):
            return [dict(zip(self.keys(), data_i)) for data_i in self._data]  # type: ignore
        else:
            msg = f"Invalid data type {type(self._data)=}."
            raise TypeError(msg)

    def to_matrix(self) -> Union[List[List], np.ndarray, Tensor]:
        if isinstance(self._data, pd.DataFrame):
            result = self._data.to_numpy()

        elif pw.isinstance_generic(self._data, List[Dict]):
            result = [list(data_i.values()) for data_i in self._data]
            if _NUMPY_AVAILABLE:
                result = np.array(result)

        elif pw.isinstance_generic(self._data, Dict[Any, pw.SupportsGetitemLen]):
            result = [[self._data[k][i] for k in self.keys()] for i in range(len(self))]
            if _NUMPY_AVAILABLE:
                result = np.array(result)

        elif isinstance(self._data, DynamicItemDataset):
            result = [list(self._data[i].values()) for i in range(len(self))]
            if _NUMPY_AVAILABLE:
                result = np.array(result)

        elif isinstance(self._data, (np.ndarray, Tensor)):
            return self._data

        else:
            msg = f"Invalid data type {type(self._data)=}."
            raise TypeError(msg)

        return result

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

        elif pw.isinstance_generic(self._data, List[Dict]):
            if len(self._data) == 0:
                return DynamicItemDataset({})

            keys = list(self._data[0].keys())
            if id_column_key is not None and id_column_key in keys:
                ids = [data_i[id_column_key] for data_i in self._data]
            else:
                ids = list(range(len(self._data)))

            did_data = {id_: data_i for id_, data_i in zip(ids, self._data)}
            return DynamicItemDataset(did_data)

        elif pw.isinstance_generic(self._data, Dict[Any, pw.SupportsGetitemIterLen]):
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
            msg = f"Invalid data type {type(self._data)=}."
            raise TypeError(msg)

    def unique(
        self,
        column_key: T_ColumnKey,
    ) -> Union[np.ndarray, Tensor, List[T_Value]]:
        col_data = self[column_key]

        if isinstance(col_data, np.ndarray):
            uniq = np.unique(col_data)
        elif isinstance(col_data, Tensor):
            uniq = col_data.unique()
        else:
            uniq = list(dict.fromkeys(col_data))

        return uniq

    @overload
    def __getitem__(
        self,
        indexer: int,
        /,
    ) -> T_Item: ...

    @overload
    def __getitem__(
        self,
        indexer: Union[MultiRows, SingleColumn, MultiColumns, tuple],
        /,
    ) -> Any: ...

    def __getitem__(
        self,
        indexer: Union[
            SingleIndexer,
            MultiIndexer,
            Tuple,
        ],
    ) -> Any:
        row_indexer: RowIndexer
        col_indexer: ColumnIndexer

        if isinstance(indexer, tuple):
            if not (
                len(indexer) >= 2
                and is_row_indexer(indexer[0])
                and is_column_indexer(indexer[1])
            ):
                raise ValueError
            row_indexer, col_indexer, *sub_indexer = indexer  # type: ignore

        elif is_row_indexer(indexer):
            row_indexer = indexer
            col_indexer = list(self.keys())  # type: ignore
            sub_indexer = ()
        else:
            row_indexer = slice(None)
            col_indexer = indexer
            sub_indexer = ()

        del indexer

        if col_indexer is None:
            col_indexer = list(self.keys())  # type: ignore
        elif isinstance(col_indexer, slice):
            col_indexer = list(self.keys()[col_indexer])  # type: ignore
        elif is_mask(col_indexer):
            keys = self.keys()
            col_indexer = tw.multihot_to_multi_indices(col_indexer)
            col_indexer = [keys[col_indexer_i] for col_indexer_i in col_indexer]  # type: ignore

        sub_indexer = tuple(sub_indexer)

        if is_single_column(col_indexer):
            if col_indexer in self.static_keys:
                return self._get_static_items(row_indexer, col_indexer, sub_indexer)
            elif col_indexer in self.dynamic_keys:
                return self._get_dynamic_items(row_indexer, col_indexer, sub_indexer)
            else:
                raise KeyError

        elif is_multi_indices(col_indexer) or is_multi_names(col_indexer):
            static_cols = [
                col_indexer_i
                for col_indexer_i in col_indexer
                if col_indexer_i in self.static_keys
            ]
            dynamic_cols = [
                col_indexer_i
                for col_indexer_i in col_indexer
                if col_indexer_i in self.dynamic_keys
            ]
            if len(static_cols) > 0 and len(dynamic_cols) == 0:
                return self._get_static_items(row_indexer, col_indexer, sub_indexer)
            elif len(static_cols) == 0 and len(dynamic_cols) > 0:
                return self._get_dynamic_items(row_indexer, col_indexer, sub_indexer)
            else:
                result = {}
                for col_indexer_i in col_indexer:
                    if col_indexer_i in self.static_keys:
                        result_i = self._get_static_items(
                            row_indexer, col_indexer, sub_indexer
                        )
                    elif col_indexer_i in self.dynamic_keys:
                        result_i = self._get_dynamic_items(
                            row_indexer, col_indexer, sub_indexer
                        )
                    else:
                        raise ValueError
                    result[col_indexer_i] = result_i
                return pw.dict_list_to_list_dict(result, "same")

        else:
            msg = f"Invalid argument type {type(col_indexer)=}"
            raise TypeError(msg)

    def _get_static_items(
        self,
        row_indexer: RowIndexer,
        col_indexer: Union[SingleIndexer, MultiIndexer],
        sub_indexer: tuple,
    ) -> Any:
        if isinstance(self._data, pd.DataFrame):
            row_indexer = tw.to_ndarray(row_indexer)
            return self._data[col_indexer][row_indexer][sub_indexer]

        elif pw.isinstance_generic(self._data, List[Dict]):
            if len(sub_indexer) != 0:
                raise NotImplementedError(f"{sub_indexer=}")

            if is_single_index(row_indexer) or isinstance(row_indexer, slice):
                row_indexer = tw.as_builtin(row_indexer)
                result = self._data[row_indexer]  # type: ignore
            elif is_multi_indices(row_indexer):
                row_indexer = tw.as_builtin(row_indexer)
                result = [self._data[index_i] for index_i in row_indexer]  # type: ignore
            elif is_mask(row_indexer):
                row_indexer = tw.multihot_to_multi_indices(row_indexer)
                result = [self._data[index_i] for index_i in row_indexer]  # type: ignore
            else:
                raise TypeError

            if is_single_column(col_indexer):
                result = [sample[col_indexer] for sample in result]
            elif is_multi_indices(col_indexer) or is_multi_names(col_indexer):
                result = [
                    {col: sample[col] for col in col_indexer} for sample in result
                ]
            else:
                msg = f"Invalid argument type {type(col_indexer)=}"
                raise TypeError(msg)

            return result

        elif pw.isinstance_generic(self._data, Dict[str, list]):
            if len(sub_indexer) != 0:
                raise NotImplementedError(f"{sub_indexer=}")

            if is_single_indexer(col_indexer):
                result = self._data[col_indexer]  # type: ignore

                if is_single_index(row_indexer) or isinstance(row_indexer, slice):
                    result = result[row_indexer]  # type: ignore
                elif is_multi_indices(row_indexer):
                    result = [result[idx] for idx in row_indexer]
                elif is_mask(row_indexer):
                    row_indexer = tw.multihot_to_multi_indices(row_indexer)
                    result = [result[idx] for idx in row_indexer]
                else:
                    msg = f"Invalid argument type {type(col_indexer)=}"
                    raise TypeError(msg)
                return result

            elif is_multi_indices(col_indexer) or is_multi_names(col_indexer):
                result = {col: self._data[col] for col in col_indexer}  # type: ignore

                if is_single_index(row_indexer):
                    result = {k: v[row_indexer] for k, v in result.items()}  # type: ignore
                    return result

                elif isinstance(row_indexer, slice):
                    result = {k: v[row_indexer] for k, v in result.items()}  # type: ignore
                elif is_multi_indices(row_indexer):
                    row_indexer = tw.as_builtin(row_indexer)
                    result = {
                        k: [v[idx] for idx in row_indexer] for k, v in result.items()
                    }
                elif is_mask(row_indexer):
                    row_indexer = tw.multihot_to_multi_indices(row_indexer)
                    result = {
                        k: [v[idx] for idx in row_indexer] for k, v in result.items()
                    }
                else:
                    msg = f"Invalid argument type {type(row_indexer)=}"
                    raise TypeError(msg)

                result = pw.dict_list_to_list_dict(result, "same")
                return result

            else:
                msg = f"Invalid argument type {type(col_indexer)=}"
                raise TypeError(msg)

        elif pw.isinstance_generic(self._data, DynamicItemDataset):
            if len(sub_indexer) != 0:
                raise NotImplementedError(f"{sub_indexer=}")

            if is_single_index(row_indexer):
                result = self._data[row_indexer]
                if is_single_indexer(col_indexer):
                    return result[col_indexer]
                elif is_multi_indices(col_indexer) or is_multi_columns(col_indexer):
                    return {k: result[k] for k in col_indexer}  # type: ignore
                else:
                    msg = f"Invalid argument type {type(col_indexer)=}"
                    raise TypeError(msg)

            elif isinstance(row_indexer, slice):
                row_indexer = list(
                    range(row_indexer.start, row_indexer.stop, row_indexer.step)
                )
                result = [self._data[idx] for idx in row_indexer]
            elif is_multi_indices(row_indexer):
                result = [self._data[idx] for idx in row_indexer]
            elif is_mask(row_indexer):
                row_indexer = tw.multihot_to_multi_indices(row_indexer)
                result = [self._data[idx] for idx in row_indexer]
            else:
                msg = f"Invalid argument type {type(row_indexer)=}"
                raise TypeError(msg)

            if is_single_indexer(col_indexer):
                return [result_i[col_indexer] for result_i in result]
            elif is_multi_indices(col_indexer) or is_multi_columns(col_indexer):
                return [{k: result_i[k] for k in col_indexer} for result_i in result]  # type: ignore
            else:
                msg = f"Invalid argument type {type(col_indexer)=}"
                raise TypeError(msg)

        elif pw.isinstance_generic(self._data, (Tensor, np.ndarray)):
            return self._data.__getitem__(
                (row_indexer, col_indexer) + sub_indexer  # type: ignore
            )

        else:
            msg = f"Invalid data type {type(self._data)=}."
            raise TypeError(msg)

    def _get_dynamic_items(
        self,
        row_indexer: RowIndexer,
        col_indexer: Union[SingleIndexer, MultiIndexer],
        sub_indexer: tuple,
    ) -> Any:
        if len(sub_indexer) != 0:
            raise ValueError

        if is_single_column(col_indexer):
            for gives, provides, fn, _ in self._dynamic_fns:
                idx = pw.find(col_indexer, provides)
                if idx == -1:
                    continue

                input_values = self[row_indexer, gives]
                output_values = []
                for in_i in input_values:
                    out_i = fn(in_i)
                    out_i = out_i[idx]
                    output_values.append(out_i)
                return output_values

        elif is_multi_indices(col_indexer) or is_multi_names(col_indexer):
            result = {}

            for col_indexer_i in col_indexer:
                for gives, provides, fn, _ in self._dynamic_fns:
                    idx = pw.find(col_indexer_i, provides)
                    if idx == -1:
                        continue

                    input_values = self[row_indexer, gives]
                    output_values = []
                    for in_i in input_values:
                        out_i = fn(in_i)
                        out_i = out_i[idx]
                        output_values.append(out_i)
                    result[col_indexer_i] = output_values
            return pw.dict_list_to_list_dict(result, "same")

        else:
            raise TypeError

    def __len__(self) -> int:
        return self.num_rows

    def __repr__(self) -> str:
        repr_ = (
            f"{self.__class__.__name__}(num_rows={self.num_rows}, keys={self.keys()})"
        )
        return repr_


@overload
def _get_static_keys(
    data: pd.DataFrame,
) -> Tuple[str, ...]: ...


@overload
def _get_static_keys(
    data: Union[
        List[Dict[T_ColumnKey, Any]],
        Dict[T_ColumnKey, List],
    ],
) -> Tuple[T_ColumnKey, ...]: ...


@overload
def _get_static_keys(
    data: Union[
        np.ndarray,
        Tensor,
    ],
) -> Tuple[int, ...]: ...


def _get_static_keys(
    data: Union[
        pd.DataFrame,
        List[Dict[T_ColumnKey, Any]],
        Dict[T_ColumnKey, List],
        DynamicItemDataset,
        np.ndarray,
        Tensor,
        TabularDataset,
    ],
) -> tuple:
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
    elif isinstance(data, (np.ndarray, Tensor)):
        return tuple(range(data.shape[1]))
    elif isinstance(data, TabularDataset):
        return data.static_keys
    else:
        msg = f"Invalid data type {type(data)=}."
        raise TypeError(msg)


def _get_dynamic_keys(
    data: Union[
        pd.DataFrame,
        List[Dict[T_ColumnKey, Any]],
        Dict[T_ColumnKey, List],
        DynamicItemDataset,
        np.ndarray,
        Tensor,
    ],
) -> Tuple[T_ColumnKey, ...]:
    if pw.isinstance_generic(
        data, (pd.DataFrame, List[Dict], dict, np.ndarray, Tensor)
    ):
        return ()
    elif isinstance(data, DynamicItemDataset):
        return tuple(data.pipeline.output_mapping.keys())
    else:
        msg = f"Invalid data type {type(data)=}."
        raise TypeError(msg)
