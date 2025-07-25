#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import os.path as osp
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    Hashable,
    List,
    Mapping,
    Optional,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)

import torch
from pythonwrench.hashlib import HashName, hash_file
from pythonwrench.warnings import warn_once
from torch import Tensor
from typing_extensions import NotRequired

from torchwrench.core.make import DeviceLike, as_device
from torchwrench.serialization.json import dump_json, load_json
from torchwrench.serialization.load_fn import LOAD_FNS, LoadFnLike, load_torch

from .paths import get_cache_dir

T_Hashable = TypeVar("T_Hashable", bound=Hashable)

pylog = logging.getLogger(__name__)


class RegistryEntry(TypedDict):
    url: str
    fname: str
    hash_value: NotRequired[str]
    hash_type: NotRequired[HashName]
    state_dict_key: NotRequired[str]
    architecture: NotRequired[str]


class RegistryHub(Generic[T_Hashable]):
    def __init__(
        self,
        infos: Mapping[T_Hashable, RegistryEntry],
        register_root: Union[str, Path] = "~/.cache/torch/hub/checkpoints",
    ) -> None:
        """
        Args:
            infos: Maps model_name to their checkpoint information, with download url, filename, hash value, hash type and state_dict key.
            register_root: Directory where checkpoints are saved. defaults to `~/.cache/torch/hub/checkpoints`.
        """
        infos = dict(infos.items())
        if register_root is None:
            register_root = get_cache_dir().joinpath("checkpoints")
        else:
            register_root = Path(register_root).resolve().expanduser()

        super().__init__()
        self._infos = infos
        self._register_root = register_root

    @property
    def infos(self) -> Dict[T_Hashable, RegistryEntry]:
        return self._infos

    @property
    def register_root(self) -> Path:
        return self._register_root.resolve()

    @property
    def names(self) -> List[T_Hashable]:
        return list(self._infos.keys())

    @property
    def paths(self) -> List[Path]:
        return [self.get_path(model_name) for model_name in self.names]

    def get_path(self, name: T_Hashable) -> Path:
        """Returns the expected filepath of an element."""
        if name not in self.names:
            msg = f"Invalid argument {name=}. (expected one of {self.names})"
            raise ValueError(msg)

        fname = self._infos[name]["fname"]
        fpath = self.register_root.joinpath(fname)
        return fpath

    def load_state_dict(
        self,
        name_or_path: Union[T_Hashable, str, Path],
        *,
        device: DeviceLike = None,
        offline: bool = False,
        load_fn: LoadFnLike = load_torch,
        load_kwds: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
    ) -> Dict[str, Tensor]:
        """Load state_dict weights.

        Args:
            model_name_or_path: Model name (case sensitive) or path to checkpoint file.
            device: Device of checkpoint weights. (deprecated)
            offline: If False, the checkpoint from a model name will be automatically downloaded.
            load_fn: Load function backend. defaults to torch.load.
            load_kwds: Optional keywords arguments passed to load_fn. defaults to None.
            verbose: Verbose level. defaults to 0.

        Returns:
            Loaded file content.
        """
        if isinstance(load_fn, str):
            if load_fn not in LOAD_FNS:
                msg = f"Invalid argument {load_fn=}. (expected one of {tuple(LOAD_FNS.keys())})"
                raise ValueError(msg)
            load_fn = LOAD_FNS[load_fn]

        if load_kwds is None:
            load_kwds = {}

        if device is not None:
            src_device = device
            device = as_device(device)
            msg = f"Deprecated argument device={src_device}. Use `load_kwds=dict(map_location={device})` with function torch.load instead."
            warn_once(msg)

            if device is not None:
                load_kwds["map_location"] = device

        if isinstance(name_or_path, (str, Path)) and osp.isfile(name_or_path):
            path = Path(name_or_path)
            name = self._get_name(path)
        else:
            name = name_or_path
            try:
                path = self.get_path(name_or_path)  # type: ignore
            except ValueError:
                msg = f"Invalid argument {name_or_path=}. (expected a path to a checkpoint file or a model name in {self.names})"
                raise ValueError(msg)

            if path.is_file():
                pass
            elif offline:
                msg = f"Cannot find checkpoint model file in '{path}' for model '{name}' with mode {offline=}."
                raise FileNotFoundError(msg)
            else:
                self.download_file(name, verbose=verbose)  # type: ignore

        del name_or_path

        info = self._infos.get(name, {})  # type: ignore
        state_dict_key = info.get("state_dict_key", None)
        data = load_fn(path, **load_kwds)

        if state_dict_key is None:
            result = data
        else:
            result = data[state_dict_key]

        if verbose >= 1:
            msg = f"Loading encoder weights from '{path}'..."
            pylog.info(msg)

        return result

    def download_file(
        self,
        name: T_Hashable,
        force: bool = False,
        check_hash: bool = True,
        verbose: int = 0,
    ) -> Tuple[Path, bool]:
        """Download checkpoint file."""
        model_path = self.get_path(name)
        exists = model_path.exists()

        if exists and not force:
            return model_path, False

        if exists and force:
            os.remove(model_path)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        url = self._infos[name]["url"]
        torch.hub.download_url_to_file(url, str(model_path), progress=verbose >= 1)

        if not check_hash:
            return model_path, True

        valid = self.is_valid_hash(name)

        if valid:
            return model_path, True
        else:
            raise ValueError(f"Invalid hash for file '{model_path}'.")

    def remove_file(
        self,
        name: T_Hashable,
    ) -> None:
        path = self.get_path(name)
        if path.is_file():
            os.remove(path)
        elif path.exists():
            msg = f"Invalid argument {name=}, which redirect to a non-file {path=}."
            raise ValueError(msg)

    def is_valid_hash(
        self,
        name: T_Hashable,
    ) -> bool:
        """Returns True if target file hash is valid. If no hash is provided in infos, this function also returns True."""
        info = self.infos[name]
        if "hash_type" not in info or "hash_value" not in info:
            msg = f"Cannot check hash for {name}. (cannot find any expected hash value or type)"
            pylog.warning(msg)
            return True

        hash_type = info["hash_type"]
        expected_hash_value = info["hash_value"]

        model_path = self.get_path(name)
        hash_value = hash_file(model_path, hash_type)
        valid = hash_value == expected_hash_value
        return valid

    def save(self, path: Union[str, Path]) -> None:
        """Save info to JSON file."""
        args = {
            "infos": self._infos,
            "register_root": str(self._register_root),
        }
        dump_json(args, path)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "RegistryHub":
        """Load register info from JSON file."""
        args = load_json(path)
        return RegistryHub(**args)

    def _get_name(self, path: Union[str, Path]) -> Optional[T_Hashable]:
        path_to_name = {
            path_i.resolve().expanduser(): name_i
            for path_i, name_i in zip(self.paths, self.names)
        }
        path = Path(path).resolve().expanduser()
        name = path_to_name.get(path, None)
        return name
