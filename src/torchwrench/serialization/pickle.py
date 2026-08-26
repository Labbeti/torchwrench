#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pythonwrench as pw
from pythonwrench.pickle import (  # noqa: F401
    load_pickle,
    loads_pickle,
    read_pickle,
)

from torchwrench.utils.pkg_hooks import register_hooks


@pw.function_alias(pw.dump_pickle, pre_fn=register_hooks)
def dump_pickle(*args, **kwargs): ...


@pw.function_alias(pw.dumps_pickle, pre_fn=register_hooks)
def dumps_pickle(*args, **kwargs): ...


@pw.function_alias(pw.save_pickle, pre_fn=register_hooks)
def save_pickle(*args, **kwargs): ...
