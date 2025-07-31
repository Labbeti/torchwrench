#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any

from pythonwrench.importlib import Placeholder


class DataFrame(Placeholder):
    def __getitem__(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __setitem__(self, *args, **kwargs) -> Any:
        raise NotImplementedError


class Series(Placeholder): ...


class RangeIndex(Placeholder): ...


class Index(Placeholder): ...
