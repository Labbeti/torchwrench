#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

import torch

from torchwrench.extras.numpy import np


def set_seed(
    seed: int, cuda: bool = True, mps: bool = True, cudnn: bool = False
) -> None:
    """Set seed for random, numpy and torch packages. Optionally for CUDA, MPS and CUDNN backend."""
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
