#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import lru_cache
from typing import Dict

import torch
from pythonwrench.importlib import is_available_package
from pythonwrench.semver import Version
from pythonwrench.warnings import warn_once


def _is_available_package_catch_all_errors(package: str) -> bool:
    try:
        return is_available_package(package)
    except Exception:
        return False


@lru_cache(1)
def _cached_is_available_package_catch_all_errors(package: str) -> bool:
    return _is_available_package_catch_all_errors(package)


@lru_cache(1)
def _get_extra_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(module.__version__)
    except ImportError:
        return "not_installed"
    except (AttributeError, RuntimeError, ModuleNotFoundError):
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


def get_extra_available_dict() -> Dict[str, bool]:
    return {
        name: _cached_is_available_package_catch_all_errors(name)
        for name in _EXTRAS_PACKAGES
    }


def get_extra_version_dict() -> Dict[str, str]:
    return {name: _get_extra_version(name) for name in _EXTRAS_PACKAGES}


def colorlog_is_available() -> bool:
    return _cached_is_available_package_catch_all_errors("colorlog")


def datasets_is_available() -> bool:
    return _cached_is_available_package_catch_all_errors("datasets")


def h5py_is_available() -> bool:
    return _cached_is_available_package_catch_all_errors("h5py")


def numpy_is_available() -> bool:
    if not _cached_is_available_package_catch_all_errors("numpy"):
        return False
    np_version = Version(_get_extra_version("numpy"))

    unsupported_versions = ["2.0.0", "2.0.1", "2.0.2"]
    pin = "numpy!=" + ",!=".join(unsupported_versions)
    msg = f"Found numpy {np_version} but it is incompatible with torchwrench. Install correct version with torchwrench[numpy] or pin numpy to a different version: '{pin}'"
    warn_once(msg, UserWarning)

    return all(np_version != Version(version) for version in unsupported_versions)


def omegaconf_is_available() -> bool:
    return _cached_is_available_package_catch_all_errors("omegaconf")


def pandas_is_available() -> bool:
    return _cached_is_available_package_catch_all_errors("pandas")


def safetensors_is_available() -> bool:
    return _cached_is_available_package_catch_all_errors("safetensors")


def scipy_is_available() -> bool:
    return _cached_is_available_package_catch_all_errors("scipy")


def speechbrain_is_available() -> bool:
    return _cached_is_available_package_catch_all_errors("speechbrain")


def tensorboard_is_available() -> bool:
    return _cached_is_available_package_catch_all_errors("tensorboard")


def torchaudio_is_available() -> bool:
    return _cached_is_available_package_catch_all_errors("torchaudio")


def torchcodec_is_available() -> bool:
    return _cached_is_available_package_catch_all_errors("torchcodec")


def torchvision_is_available() -> bool:
    return _cached_is_available_package_catch_all_errors("torchvision")


def tqdm_is_available() -> bool:
    return _cached_is_available_package_catch_all_errors("tqdm")


def yaml_is_available() -> bool:
    return _cached_is_available_package_catch_all_errors("yaml")


def torch_version_ge_1_13() -> bool:
    version_str = str(torch.__version__)
    version = Version.from_str(version_str)
    return version >= Version("1.13.0")
