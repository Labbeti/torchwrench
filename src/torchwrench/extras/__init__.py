#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .numpy import _NUMPY_AVAILABLE
    from .pandas import _PANDAS_AVAILABLE
    from .speechbrain import _SPEECHBRAIN_AVAILABLE
    from .yaml import _YAML_AVAILABLE

else:
    import lazy_loader as lazy

    __getattr__, __dir__, __all__ = lazy.attach(
        __name__,
        submodules=["numpy", "pandas", "speechbrain", "yaml"],
        submod_attrs={
            "numpy": ["NUMPY_AVAILABLE"],
            "pandas": ["_PANDAS_AVAILABLE"],
            "speechbrain": ["_SPEECHBRAIN_AVAILABLE"],
            "yaml": ["_YAML_AVAILABLE"],
        },
    )

del TYPE_CHECKING
