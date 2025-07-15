#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import wraps
from typing import Callable, Final, Iterable, TypeVar, Union

import torch
from pythonwrench.importlib import is_available_package
from pythonwrench.semver import Version
from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


def _get_extra_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(module.__version__)
    except ImportError:
        return "not_installed"
    except AttributeError:
        return "unknown"


_EXTRAS_PACKAGES = (
    "colorlog",
    "h5py",
    "numpy",
    "omegaconf",
    "pandas",
    "safetensors",
    "scipy",
    "tensorboard",
    "torchaudio",
    "tqdm",
    "yaml",
)
_EXTRA_AVAILABLE = {name: is_available_package(name) for name in _EXTRAS_PACKAGES}
_EXTRA_VERSION = {name: _get_extra_version(name) for name in _EXTRAS_PACKAGES}


_COLORLOG_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["colorlog"]
_H5PY_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["h5py"]
_NUMPY_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["numpy"]
_OMEGACONF_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["omegaconf"]
_PANDAS_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["pandas"]
_SAFETENSORS_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["safetensors"]
_SCIPY_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["scipy"]
_TENSORBOARD_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["tensorboard"]
_TORCHAUDIO_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["torchaudio"]
_TQDM_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["tqdm"]
_YAML_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["yaml"]


def requires_packages(
    arg0: Union[Callable[P, T], Iterable[str], str] = ...,
    /,
    *args: str,
) -> Callable:
    """Decorator to wrap a function and raises an error if the function is called.

    ```
    >>> @requires_packages("pandas")
    >>> def f(x):
    >>>     return x
    >>> f(1)  # raises ImportError if pandas is not installed
    ```
    """
    if arg0 is ...:
        packages = []
    elif callable(arg0):
        packages = args
    elif isinstance(arg0, str):
        packages = [arg0] + list(args)
    elif isinstance(arg0, Iterable):
        packages = list(arg0) + list(args)

    else:
        raise TypeError

    def _wrap(fn: Callable[P, T]) -> Callable[P, T]:
        @wraps(fn)
        def _impl(*args: P.args, **kwargs: P.kwargs) -> T:
            missing = [pkg for pkg in packages if not is_available_package(pkg)]
            if len(missing) == 0:
                return fn(*args, **kwargs)
            else:
                prefix = "\n - "
                missing_str = prefix.join(missing)
                msg = (
                    f"Cannot use/import objects because the following optionals dependencies are missing:"
                    f"{prefix}{missing_str}\n"
                    f"Please install them using `pip install torchwrench[extras]`."
                )
                raise ImportError(msg)

        return _impl

    return _wrap


def torch_version_ge_1_13() -> bool:
    version_str = str(torch.__version__)
    version = Version.from_str(version_str)
    return version >= Version("1.13.0")
