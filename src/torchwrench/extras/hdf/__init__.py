#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .common import h5py_is_available
    from .dataset import HDFDataset  # noqa: E402
    from .pack import pack_to_hdf  # noqa: E402

else:
    import lazy_loader as lazy

    __getattr__, __dir__, __all__ = lazy.attach(
        __name__,
        submodules=["common", "dataset", "pack"],
        submod_attrs={
            "common": ["h5py_is_available"],
            "dataset": ["HDFDataset"],
            "pack": ["pack_to_hdf"],
        },
    )


del TYPE_CHECKING
