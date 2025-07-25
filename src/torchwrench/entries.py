#!/usr/bin/env python
# -*- coding: utf-8 -*-

import platform
import sys
from pathlib import Path
from typing import Dict, Union

import torch
from pythonwrench.json import dump_json

import torchwrench as tw
from torchwrench.core.packaging import _EXTRA_VERSION
from torchwrench.hub.paths import get_cache_dir, get_dir, get_tmp_dir
from torchwrench.utils.data.dataloader import get_auto_num_cpus, get_auto_num_gpus


def get_package_repository_path() -> str:
    """Return the absolute path where the source code of this package is installed."""
    return str(Path(__file__).parent.parent.parent)


def get_install_info() -> Dict[str, Union[str, int]]:
    install_info = {
        "torchwrench": tw.__version__,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "os": platform.system(),
        "architecture": platform.architecture()[0],
        "torch": str(torch.__version__),
        "num_cpus": get_auto_num_cpus(),
        "num_gpus": get_auto_num_gpus(),
        "package_path": get_package_repository_path(),
        "tmpdir": str(get_tmp_dir()),
        "cachedir": str(get_cache_dir()),
        "torch_hub": get_dir(),
    }
    install_info.update({k: str(v) for k, v in _EXTRA_VERSION.items()})
    return install_info


def print_install_info() -> None:
    """Show main packages versions."""
    install_info = get_install_info()
    dumped = dump_json(install_info)
    print(dumped)


if __name__ == "__main__":
    print_install_info()
