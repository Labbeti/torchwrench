#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pythonwrench as pw


def register_hooks(*args, **kwargs) -> None:
    from torchwrench.nn.functional.checksum import checksum_any  # noqa: F401
    from torchwrench.serialization.common import as_builtin  # noqa: F401


@pw.function_alias(pw.disk_cache_call, pre_fn=register_hooks)
def disk_cache_call(*args, **kwargs): ...


@pw.function_alias(pw.disk_cache_decorator, pre_fn=register_hooks)
def disk_cache_decorator(*args, **kwargs): ...
