#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import math

import pythonwrench as pw
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer

pylog = logging.getLogger(__name__)


class CosDecayScheduler(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        n_steps: int,
        last_epoch: int = -1,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            lr_lambda=CosDecayRule(n_steps),
            last_epoch=last_epoch,
        )


class CosDecayRule:
    # Note : use class instead of function for scheduler rules to become pickable for multiple-GPU with Lightning
    def __init__(self, n_steps: int) -> None:
        __check_args(n_steps)
        super().__init__()
        self.n_steps = n_steps

    def __call__(self, step: int) -> float:
        return cos_decay_rule(step, self.n_steps)


def cos_decay_rule(step: int, n_steps: int) -> float:
    __check_args(n_steps)
    step = min(step, n_steps - 1)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * step / n_steps))
    return coefficient


def __check_args(n_steps: int) -> None:
    if n_steps < 0:
        msg = f"Invalid argument {n_steps=} < 0 for cos decay."
        raise ValueError(msg)
    elif n_steps == 0:
        msg = f"Replacing {n_steps=} by n_steps=1 for cos decay."
        pw.warn_once(msg)
        n_steps = max(n_steps, 1)
