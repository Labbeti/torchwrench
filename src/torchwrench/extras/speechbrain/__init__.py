#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .definitions import (  # noqa: F401
        DynamicItemDataset,  # type: ignore
        speechbrain_is_available,
    )

else:
    import lazy_loader as lazy

    __getattr__, __dir__, __all__ = lazy.attach(
        __name__,
        submodules=["definitions"],
        submod_attrs={
            "definitions": ["speechbrain_is_available", "DynamicItemDataset"],
        },
    )

del TYPE_CHECKING
