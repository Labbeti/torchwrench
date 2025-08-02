#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import (
    Any,
    Iterable,
    Union,
)

import pythonwrench as pw
from typing_extensions import TypeGuard

import torchwrench as tw
from torchwrench.extras.numpy import (
    is_numpy_bool_array,
    is_numpy_integral_array,
    is_numpy_str_array,
    np,
)

SingleIndex = Union[int, np.ndarray, np.generic, tw.IntegralTensor0D]
MultiIndices = Union[Iterable[int], np.ndarray, tw.IntegralTensor1D]
Mask = Union[Iterable[bool], np.ndarray, tw.BoolTensor1D]
SingleName = Union[str, np.ndarray, np.generic]
MultiNames = Union[Iterable[str], np.ndarray]

SingleRow = SingleIndex
MultiRows = Union[MultiIndices, Mask, slice]

SingleColumn = Union[SingleIndex, SingleName]
MultiColumns = Union[MultiIndices, Mask, slice, MultiNames]

RowIndexer = Union[SingleRow, MultiRows]
ColumnIndexer = Union[SingleColumn, MultiColumns]

SingleIndexer = Union[SingleIndex, SingleName]
MultiIndexer = Union[MultiIndices, Mask, slice, MultiNames]


def is_single_index(x) -> TypeGuard[SingleIndex]:
    return pw.isinstance_generic(x, (int, tw.IntegralTensor0D)) or (
        is_numpy_integral_array(x) and x.ndim == 0
    )


def is_mask(x: Any) -> TypeGuard[Mask]:
    return (
        pw.isinstance_generic(x, Iterable[bool])
        or (is_numpy_bool_array(x) and x.ndim == 1)
        or isinstance(x, tw.BoolTensor1D)
    ) and not isinstance(x, tuple)


def is_multi_indices(x: Any) -> TypeGuard[MultiIndices]:
    return (
        pw.isinstance_generic(x, Iterable[int])
        or (is_numpy_integral_array(x) and x.ndim == 1)
        or isinstance(x, tw.IntegralTensor1D)
    ) and not isinstance(x, tuple)


def is_single_name(x: Any) -> TypeGuard[SingleName]:
    return pw.isinstance_generic(x, str) or (is_numpy_str_array(x) and x.ndim == 0)


def is_multi_names(x: Any) -> TypeGuard[MultiNames]:
    return pw.isinstance_generic(x, (Iterable[str])) or (
        is_numpy_str_array(x) and x.ndim == 1
    )


def is_single_row(x: Any) -> TypeGuard[SingleRow]:
    return is_single_index(x)


def is_multi_rows(x: Any) -> TypeGuard[MultiRows]:
    return is_multi_indices(x) or is_mask(x) or isinstance(x, slice)


def is_single_column(x: Any) -> TypeGuard[SingleColumn]:
    return is_single_index(x) or is_single_name(x)


def is_multi_columns(x: Any) -> TypeGuard[MultiColumns]:
    return (
        is_multi_indices(x) or is_mask(x) or isinstance(x, slice) or is_multi_names(x)
    )


def is_row_indexer(x: Any) -> TypeGuard[RowIndexer]:
    return is_single_row(x) or is_multi_rows(x)


def is_column_indexer(x: Any) -> TypeGuard[ColumnIndexer]:
    return is_single_column(x) or is_multi_columns(x)


def is_single_indexer(x: Any) -> TypeGuard[SingleIndexer]:
    return is_single_index(x) or is_single_name(x)


def is_multi_indexer(x: Any) -> TypeGuard[MultiIndexer]:
    return (
        is_multi_indices(x) or is_mask(x) or isinstance(x, slice) or is_multi_names(x)
    )
