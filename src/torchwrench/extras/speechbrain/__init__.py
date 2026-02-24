#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .definitions import (  # noqa: F401
        _SPEECHBRAIN_AVAILABLE,
        DynamicItemDataset,
    )

else:
    import lazy_loader as lazy

    __getattr__, __dir__, __all__ = lazy.attach(
        __name__,
        submodules=["definitions"],
        submod_attrs={
            "definitions": ["_SPEECHBRAIN_AVAILABLE", "DynamicItemDataset"],
        },
    )

    x = __getattr__("_SPEECHBRAIN_AVAILABLE")
    print(f"{x=}")


del TYPE_CHECKING
