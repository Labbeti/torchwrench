#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any, Optional, Union

from pythonwrench.json import dump_json as _dump_json_base
from pythonwrench.json import load_json  # noqa: F401

from .common import as_builtin


def dump_json(
    data: Any,
    fpath: Union[str, Path, None] = None,
    *,
    overwrite: bool = True,
    make_parents: bool = True,
    to_builtins: bool = False,
    # JSON dump kwargs
    indent: Optional[int] = 4,
    ensure_ascii: bool = False,
    **json_dump_kwds,
) -> str:
    """Dump content to JSON format into a string and/or file.

    Args:
        data: Data to dump to JSON.
        fpath: Optional filepath to save dumped data. Not used if None. defaults to None.
        overwrite: If True, overwrite target filepath. defaults to True.
        make_parents: Build intermediate directories to filepath. defaults to True.
        to_builtins: Convert data to built-in equivalent. defaults to False.
        indent: JSON indentation size in spaces. defaults to 4.
        ensure_ascii: Ensure only ASCII characters. defaults to False.
        **json_dump_kwds: Other `json.dump` args.

    Returns:
        Dumped content as string.
    """
    if to_builtins:
        data = as_builtin(data)

    return _dump_json_base(
        data,
        fpath,
        overwrite=overwrite,
        make_parents=make_parents,
        indent=indent,
        ensure_ascii=ensure_ascii,
        **json_dump_kwds,
    )
