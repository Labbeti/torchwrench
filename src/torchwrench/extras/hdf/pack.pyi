from pathlib import Path
from typing import Any, Callable, Generic, Hashable, Literal, Mapping, TypeVar

import numpy as np
from _typeshed import Incomplete
from pythonwrench.typing import SupportsGetitemLen, SupportsIterLen
from typing_extensions import TypeAlias

from torchwrench import nn as nn
from torchwrench.extras.hdf.common import (
    HDF_ENCODING as HDF_ENCODING,
)
from torchwrench.extras.hdf.common import (
    HDF_STRING_DTYPE as HDF_STRING_DTYPE,
)
from torchwrench.extras.hdf.common import (
    HDF_VOID_DTYPE as HDF_VOID_DTYPE,
)
from torchwrench.extras.hdf.common import (
    SHAPE_SUFFIX as SHAPE_SUFFIX,
)
from torchwrench.extras.hdf.common import (
    ExistsMode as ExistsMode,
)
from torchwrench.extras.hdf.common import (
    HDFItemType as HDFItemType,
)
from torchwrench.extras.hdf.dataset import HDFDataset as HDFDataset
from torchwrench.extras.numpy import (
    merge_numpy_dtypes as merge_numpy_dtypes,
)
from torchwrench.extras.numpy import (
    numpy_is_complex_dtype as numpy_is_complex_dtype,
)
from torchwrench.extras.numpy import (
    scan_shape_dtypes as scan_shape_dtypes,
)
from torchwrench.types import BuiltinScalar as BuiltinScalar

K = TypeVar("K", covariant=True, bound=Hashable)
V = TypeVar("V", covariant=True)
T = TypeVar("T", covariant=True)
T_DictOrTuple = TypeVar("T_DictOrTuple", tuple, dict, covariant=True)
HDFDType: TypeAlias
pylog: Incomplete

def pack_to_hdf(
    dataset: SupportsGetitemLen[T_DictOrTuple]
    | SupportsIterLen[T_DictOrTuple]
    | Mapping[str, SupportsGetitemLen],
    hdf_fpath: str | Path,
    pre_transform: Callable[[T_DictOrTuple], T_DictOrTuple] | None = ...,
    *,
    batch_size: int = 32,
    num_workers: int | Literal["auto"] = "auto",
    skip_scan: bool = False,
    encoding: str = ...,
    file_kwds: dict[str, Any] | None = None,
    col_kwds: dict[str, Any] | None = None,
    shape_suffix: str = ...,
    store_str_as_vlen: bool = False,
    user_attrs: Any = None,
    exists: ExistsMode = "error",
    ds_kwds: dict[str, Any] | None = None,
    verbose: int = 0,
) -> HDFDataset[T_DictOrTuple, T_DictOrTuple]: ...
def hdf_dtype_to_fill_value(hdf_dtype: HDFDType | None) -> BuiltinScalar: ...
def numpy_dtype_to_hdf_dtype(
    dtype: np.dtype | None, *, encoding: str = ...
) -> np.dtype: ...
def hdf_dtype_to_numpy_dtype(hdf_dtype: HDFDType) -> np.dtype: ...

class _DictWrapper(Generic[K, V]):
    mapping: Incomplete
    def __init__(self, mapping: Mapping[K, SupportsGetitemLen[V]]) -> None: ...
    def __getitem__(self, index: int) -> dict[K, V]: ...
    def __len__(self) -> int: ...
