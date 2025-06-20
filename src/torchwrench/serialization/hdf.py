#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchwrench.extras.hdf import HDFDataset, pack_to_hdf
from pythonwrench.functools import function_alias


@function_alias(pack_to_hdf)
def dump_hdf(*args, **kwargs):
    ...


@function_alias(HDFDataset)
def load_hdf(*args, **kwargs):
    ...
