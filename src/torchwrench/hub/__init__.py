#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING

from torch.hub import (
    download_url_to_file,
    get_dir,
    help,
    list,
    load,
    load_state_dict_from_url,
    set_dir,
)

if TYPE_CHECKING:
    from .download import download_file
    from .paths import get_cache_dir, get_tmp_dir
    from .registry import RegistryEntry, RegistryHub

else:
    import lazy_loader as lazy

    __getattr__, __dir__, __all__ = lazy.attach(
        __name__,
        submodules=["download", "paths", "registry"],
        submod_attrs={
            "download": ["download_file"],
            "paths": ["get_cache_dir", "get_tmp_dir"],
            "registry": ["RegistryEntry", "RegistryHub"],
        },
    )
