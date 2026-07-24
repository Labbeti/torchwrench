#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import data as data
    from . import disk_cache as disk_cache
    from . import rng as rng

else:
    import lazy_loader as lazy

    __getattr__, __dir__, _ = lazy.attach(
        __name__,
        submodules=[
            "disk_cache",
            "data",
            "rng",
        ],
        submod_attrs={},
    )

del TYPE_CHECKING
