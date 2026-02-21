#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING

from pythonwrench.jsonl import dump_jsonl, load_jsonl
from torch.serialization import (
    LoadEndianness,
    SourceChangeWarning,
    StorageType,
    check_module_version_greater_or_equal,
    default_restore_location,
    get_default_load_endianness,
    load,
    location_tag,
    mkdtemp,
    normalize_storage_type,
    register_package,
    save,
    set_default_load_endianness,
    storage_to_tensor_type,
    validate_cuda_device,
    validate_hpu_device,
)

if TYPE_CHECKING:
    from .common import as_builtin
    from .csv import dump_csv, load_csv
    from .dump_fn import dump_to, save_to
    from .json import dump_json, load_json
    from .load_fn import load_from, read_from
    from .pickle import dump_pickle, load_pickle
    from .torch import dump_torch, load_torch

else:
    import lazy_loader as lazy

    __getattr__, __dir__, __all__ = lazy.attach(
        __name__,
        submodules=[],
        submod_attrs={
            "common": ["as_builtin"],
            "csv": ["dump_csv", "load_csv"],
            "dump_fn": ["dump_to", "save_to"],
            "json": ["dump_json", "loadload_json_csv"],
            "load_fn": ["load_from", "read_from"],
            "pickle": ["dump_pickle", "load_pickle"],
            "torch": ["dump_torch", "load_torch"],
        },
    )

del TYPE_CHECKING
