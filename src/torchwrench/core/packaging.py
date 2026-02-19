#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Final

import torch
from pythonwrench.importlib import is_available_package
from pythonwrench.semver import Version


def _get_extra_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(module.__version__)
    except ImportError:
        return "not_installed"
    except AttributeError:
        return "unknown"


EXTRA_PACKAGES = (
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
    "tqdm",
    "yaml",
)
EXTRA_AVAILABLE = {name: is_available_package(name) for name in EXTRA_PACKAGES}
EXTRA_VERSION = {name: _get_extra_version(name) for name in EXTRA_PACKAGES}


COLORLOG_AVAILABLE: Final[bool] = EXTRA_AVAILABLE["colorlog"]
DATASETS_AVAILABLE: Final[bool] = EXTRA_AVAILABLE["datasets"]
H5PY_AVAILABLE: Final[bool] = EXTRA_AVAILABLE["h5py"]
NUMPY_AVAILABLE: Final[bool] = EXTRA_AVAILABLE["numpy"]
OMEGACONF_AVAILABLE: Final[bool] = EXTRA_AVAILABLE["omegaconf"]
PANDAS_AVAILABLE: Final[bool] = EXTRA_AVAILABLE["pandas"]
SAFETENSORS_AVAILABLE: Final[bool] = EXTRA_AVAILABLE["safetensors"]
SCIPY_AVAILABLE: Final[bool] = EXTRA_AVAILABLE["scipy"]
SPEECHBRAIN_AVAILABLE: Final[bool] = EXTRA_AVAILABLE["speechbrain"]
TENSORBOARD_AVAILABLE: Final[bool] = EXTRA_AVAILABLE["tensorboard"]
TORCHAUDIO_AVAILABLE: Final[bool] = EXTRA_AVAILABLE["torchaudio"]
TQDM_AVAILABLE: Final[bool] = EXTRA_AVAILABLE["tqdm"]
YAML_AVAILABLE: Final[bool] = EXTRA_AVAILABLE["yaml"]


def torch_version_ge_1_13() -> bool:
    version_str = str(torch.__version__)
    version = Version.from_str(version_str)
    return version >= Version("1.13.0")
