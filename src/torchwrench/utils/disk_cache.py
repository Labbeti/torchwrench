#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pythonwrench as pw

from torchwrench.utils.pkg_hooks import register_hooks


@pw.function_alias(pw.disk_cache_call, pre_fn=register_hooks)
def disk_cache_call(*args, **kwargs): ...


@pw.function_alias(pw.disk_cache_decorator, pre_fn=register_hooks)
def disk_cache_decorator(*args, **kwargs): ...
