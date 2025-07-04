#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module used for typing when numpy is not installed."""

from typing import Any

from pythonwrench.importlib import Placeholder
from pythonwrench.inspect import get_current_fn_name


class generic(Placeholder): ...


class number(generic): ...


class bool_(generic): ...


class dtype(Placeholder): ...


class ndarray(Placeholder):
    def __getitem__(self, *args) -> Any:
        return self


def array(x: Any, *args, **kwargs):
    msg = f"Cannot call function '{get_current_fn_name()}' because optional dependency 'numpy' is not installed. Please install it using 'pip install torchwrench[extras]'"
    raise NotImplementedError(msg)


def asarray(x: Any, *args, **kwargs):
    msg = f"Cannot call function '{get_current_fn_name()}' because optional dependency 'numpy' is not installed. Please install it using 'pip install torchwrench[extras]'"
    raise NotImplementedError(msg)


def iscomplexobj(x: Any):
    msg = f"Cannot call function '{get_current_fn_name()}' because optional dependency 'numpy' is not installed. Please install it using 'pip install torchwrench[extras]'"
    raise NotImplementedError(msg)


def empty(*args, **kwargs):
    msg = f"Cannot call function '{get_current_fn_name()}' because optional dependency 'numpy' is not installed. Please install it using 'pip install torchwrench[extras]'"
    raise NotImplementedError(msg)


class complex64(dtype): ...


class complex128(dtype): ...


class complex256(dtype): ...


class float16(dtype): ...


class float32(dtype): ...


class float64(dtype): ...


class float128(dtype): ...


class floating(dtype): ...


class int16(dtype): ...


class int32(dtype): ...


class int64(dtype): ...


class uint8(dtype): ...
