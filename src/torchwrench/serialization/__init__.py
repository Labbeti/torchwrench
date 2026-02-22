#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING

from pythonwrench.jsonl import dump_jsonl, load_jsonl

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
