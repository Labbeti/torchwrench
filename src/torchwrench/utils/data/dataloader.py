#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from pythonwrench.functools import function_alias
from pythonwrench.os import get_num_cpus_available


@function_alias(torch.cuda.device_count)
def get_num_gpus_available(*args, **kwargs): ...


@function_alias(get_num_cpus_available)
def get_auto_num_cpus(*args, **kwargs): ...


@function_alias(get_num_gpus_available)
def get_auto_num_gpus(*args, **kwargs): ...
