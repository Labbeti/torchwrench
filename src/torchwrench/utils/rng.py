#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from typing import Optional

import torch

from torchwrench.extras.numpy import np


def set_seed(
    seed: Optional[int] = None,
    cuda: bool = True,
    mps: bool = True,
    cudnn: bool = False,
) -> int:
    """Set seed for random, numpy and torch packages. Optionally for CUDA, MPS and CUDNN backend."""
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed_all(seed)
    if mps:
        torch.mps.manual_seed(seed)
    if cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
