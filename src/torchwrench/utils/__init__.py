#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import data as data
    from . import disk_cache as disk_cache
    from . import monitoring as monitoring
    from . import pkg_hooks as pkg_hooks
    from . import return_types as return_types
    from . import rng as rng

else:
    import lazy_loader as lazy

    __getattr__, __dir__, _ = lazy.attach(
        __name__,
        submodules=[
            "disk_cache",
            "data",
            "monitoring",
            "pkg_hooks",
            "return_types",
            "rng",
        ],
        submod_attrs={},
    )

del TYPE_CHECKING
