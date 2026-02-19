from typing import Any, TypedDict, TypeVar

import numpy as np
from _typeshed import Incomplete

T = TypeVar("T", covariant=True)
HDFItemType: Incomplete
ExistsMode: Incomplete
HDF_ENCODING: str
SHAPE_SUFFIX: str
HDF_STRING_DTYPE: np.dtype
HDF_VOID_DTYPE: np.dtype

class HDFDatasetAttributes(TypedDict):
    added_columns: list[str]
    creation_date: str
    encoding: str
    file_kwds: dict[str, Any]
    global_hash_value: int
    info: dict[str, Any]
    item_type: HDFItemType
    length: int
    load_as_complex: dict[str, bool]
    pre_transform: str
    shape_suffix: str
    source_dataset: str
    src_np_dtypes: dict[str, np.dtype]
    use_vlen_str: bool
    user_attrs: Any
    torchwrench_version: str
