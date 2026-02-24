#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING

from torch.optim import *  # type: ignore

if TYPE_CHECKING:
    from .schedulers import CosDecayRule, CosDecayScheduler
    from .utils import create_params_groups_bias, get_lr, get_lrs

else:
    import lazy_loader as lazy

    __getattr__, __dir__, __all__ = lazy.attach(
        __package__,
        submodules=["schedulers", "utils"],
        submod_attrs={
            "schedulers": ["CosDecayScheduler", "CosDecayRule"],
            "utils": ["get_lr", "get_lrs", "create_params_groups_bias"],
        },
    )


del TYPE_CHECKING
