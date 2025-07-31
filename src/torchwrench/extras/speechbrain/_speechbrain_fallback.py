#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pythonwrench.importlib import Placeholder


class DynamicItemDataset(Placeholder):
    def __len__(self) -> int: ...
