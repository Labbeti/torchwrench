#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pythonwrench as pw
from pythonwrench.json import (  # noqa: F401
    load_json,
    loads_json,
    read_json,
)

from torchwrench.utils.pkg_hooks import register_hooks


@pw.function_alias(pw.dump_json, pre_fn=register_hooks)
def dump_json(*args, **kwargs): ...


@pw.function_alias(pw.dumps_json, pre_fn=register_hooks)
def dumps_json(*args, **kwargs): ...


@pw.function_alias(pw.save_json, pre_fn=register_hooks)
def save_json(*args, **kwargs): ...
