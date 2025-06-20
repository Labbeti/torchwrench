#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tempfile
from pathlib import Path

from torch.hub import get_dir

from pythonwrench.functools import function_alias


def get_tmp_dir(mkdir: bool = False, make_parents: bool = True) -> Path:
    """Returns torchwrench temporary directory.

    Defaults is `/tmp/torchwrench`.
    Can be overriden with 'torchwrench_TMPDIR' environment variable.
    """
    default = tempfile.gettempdir()
    result = os.getenv("torchwrench_TMPDIR", default)
    result = Path(result).joinpath("torchwrench").resolve().expanduser()
    if mkdir:
        result.mkdir(parents=make_parents, exist_ok=True)
    return result


def get_cache_dir(mkdir: bool = False, make_parents: bool = True) -> Path:
    """Returns torchwrench cache directory for storing checkpoints, data and models.

    Defaults is `~/.cache/torchwrench`.
    Can be overriden with 'torchwrench_CACHEDIR' environment variable.
    """
    default = Path.home().joinpath(".cache", "torchwrench")
    result = os.getenv("torchwrench_CACHEDIR", default)
    result = Path(result).resolve().expanduser()
    if mkdir:
        result.mkdir(parents=make_parents, exist_ok=True)
    return result


@function_alias(get_dir)
def get_torch_cache_dir(*args, **kwargs):
    ...
