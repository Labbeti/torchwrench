#!/usr/bin/env python
# -*- coding: utf-8 -*-


def register_hooks(*args, **kwargs) -> None:
    from torchwrench.nn.functional.checksum import checksum_any  # noqa: F401
    from torchwrench.serialization.common import as_builtin  # noqa: F401
