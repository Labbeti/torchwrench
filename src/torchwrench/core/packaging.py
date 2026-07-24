#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Final

import torch
from pythonwrench.importlib import is_available_package
from pythonwrench.semver import Version


def _is_available_package_catch_all_errors(package: str) -> bool:
    try:
        return is_available_package(package)
    except Exception:
        return False


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
    "datasets",
    "h5py",
    "numpy",
    "omegaconf",
    "pandas",
    "safetensors",
    "scipy",
    "speechbrain",
    "tensorboard",
    "torchaudio",
    "torchcodec",
    "torchvision",
    "tqdm",
    "yaml",
)
_EXTRA_AVAILABLE = {
    name: _is_available_package_catch_all_errors(name) for name in _EXTRAS_PACKAGES
}
_EXTRA_VERSION = {name: _get_extra_version(name) for name in _EXTRAS_PACKAGES}


_COLORLOG_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["colorlog"]
_DATASETS_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["datasets"]
_H5PY_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["h5py"]
_NUMPY_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["numpy"]
_OMEGACONF_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["omegaconf"]
_PANDAS_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["pandas"]
_SAFETENSORS_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["safetensors"]
_SCIPY_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["scipy"]
_SPEECHBRAIN_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["speechbrain"]
_TENSORBOARD_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["tensorboard"]
_TORCHAUDIO_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["torchaudio"]
_TORCHCODEC_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["torchcodec"]
_TORCHVISION_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["torchvision"]
_TQDM_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["tqdm"]
_YAML_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["yaml"]


def torch_version_ge_1_13() -> bool:
    version_str = str(torch.__version__)
    version = Version.from_str(version_str)
    return version >= Version("1.13.0")
