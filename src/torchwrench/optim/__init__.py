#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING

from torch.optim._adafactor import Adafactor as Adafactor
from torch.optim.adadelta import Adadelta as Adadelta
from torch.optim.adagrad import Adagrad as Adagrad
from torch.optim.adam import Adam as Adam
from torch.optim.adamax import Adamax as Adamax
from torch.optim.adamw import AdamW as AdamW
from torch.optim.asgd import ASGD as ASGD
from torch.optim.lbfgs import LBFGS as LBFGS
from torch.optim.nadam import NAdam as NAdam
from torch.optim.optimizer import Optimizer as Optimizer
from torch.optim.radam import RAdam as RAdam
from torch.optim.rmsprop import RMSprop as RMSprop
from torch.optim.rprop import Rprop as Rprop
from torch.optim.sgd import SGD as SGD
from torch.optim.sparse_adam import SparseAdam as SparseAdam

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
