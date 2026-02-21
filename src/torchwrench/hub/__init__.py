#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.hub import (
        download_url_to_file,
        get_dir,
        help,
        list,
        load,
        load_state_dict_from_url,
        set_dir,
    )

    from .download import download_file
    from .paths import get_cache_dir, get_tmp_dir
    from .registry import RegistryEntry, RegistryHub

else:
    import lazy_loader as lazy

    hub = lazy.load("torch.hub")
    list = hub.list
    load = hub.load
    load_state_dict_from_url = hub.load_state_dict_from_url
    set_dir = hub.set_dir
    get_dir = hub.get_dir
    help = hub.help
    download_url_to_file = hub.download_url_to_file
    del hub

    __getattr__, __dir__, __all__ = lazy.attach(
        __name__,
        submodules=["download", "paths", "registry"],
        submod_attrs={
            "download": ["download_file"],
            "paths": ["get_cache_dir", "get_tmp_dir"],
            "registry": ["RegistryEntry", "RegistryHub"],
        },
    )
