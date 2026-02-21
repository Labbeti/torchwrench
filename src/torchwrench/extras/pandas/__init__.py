#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .definitions import _PANDAS_AVAILABLE, pandas, pd
else:
    import lazy_loader as lazy

    __getattr__, __dir__, __all__ = lazy.attach(
        __name__,
        submodules=["definitions"],
        submod_attrs={
            "definitions": ["pd", "_PANDAS_AVAILABLE", "pandas"],
        },
    )
