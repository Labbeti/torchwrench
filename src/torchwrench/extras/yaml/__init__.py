#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .definitions import YamlLoaders, yaml, yaml_is_available
    from .yaml import (
        IgnoreTagLoader,
        SplitTagLoader,
        dump_yaml,
        dumps_yaml,
        load_yaml,
        loads_yaml,
        read_yaml,
        save_yaml,
    )

else:
    import lazy_loader as lazy

    __getattr__, __dir__, __all__ = lazy.attach(
        __name__,
        submodules=["definitions", "yaml"],
        submod_attrs={
            "definitions": ["yaml_is_available", "YamlLoaders", "yaml"],
            "yaml": [
                "IgnoreTagLoader",
                "SplitTagLoader",
                "dump_yaml",
                "dumps_yaml",
                "load_yaml",
                "loads_yaml",
                "read_yaml",
                "save_yaml",
            ],
        },
    )

del TYPE_CHECKING
