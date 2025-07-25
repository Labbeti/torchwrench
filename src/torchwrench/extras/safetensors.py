#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union, overload

from pythonwrench.inspect import get_fullname
from pythonwrench.io import _setup_output_fpath
from pythonwrench.typing.checks import isinstance_generic
from safetensors import safe_open
from safetensors.torch import save
from torch import Tensor

from torchwrench.nn import functional as F


@overload
def load_safetensors(
    fpath: Union[str, Path],
    *,
    device: str = "cpu",
    return_metadata: Literal[False] = False,
) -> Dict[str, Tensor]: ...


@overload
def load_safetensors(
    fpath: Union[str, Path],
    *,
    device: str = "cpu",
    return_metadata: Literal[True],
) -> Tuple[Dict[str, Tensor], Dict[str, str]]: ...


def load_safetensors(
    fpath: Union[str, Path],
    *,
    device: str = "cpu",
    return_metadata: bool = False,
) -> Union[Dict[str, Tensor], Tuple[Dict[str, Tensor], Dict[str, str]]]:
    tensors = {}
    with safe_open(fpath, framework="pt", device=device) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

        if return_metadata:
            metadata = f.metadata()
            result = tensors, metadata
        else:
            result = tensors

    return result


@overload
def dump_safetensors(
    tensors: Dict[str, Tensor],
    fpath: Union[str, Path, None] = None,
    metadata: Optional[Dict[str, str]] = None,
    *,
    overwrite: bool = True,
    make_parents: bool = True,
    convert_to_tensor: Literal[False] = False,
) -> bytes: ...


@overload
def dump_safetensors(
    tensors: Dict[str, Any],
    fpath: Union[str, Path, None] = None,
    metadata: Optional[Dict[str, str]] = None,
    *,
    overwrite: bool = True,
    make_parents: bool = True,
    convert_to_tensor: Literal[True],
) -> bytes: ...


def dump_safetensors(
    tensors: Dict[str, Any],
    fpath: Union[str, Path, None] = None,
    metadata: Optional[Dict[str, str]] = None,
    *,
    overwrite: bool = True,
    make_parents: bool = True,
    convert_to_tensor: bool = False,
) -> bytes:
    """Dump tensors to safetensors format. Requires safetensors package installed."""
    if convert_to_tensor:
        tensors = {k: F.as_tensor(v) for k, v in tensors.items()}
    elif not isinstance_generic(tensors, Dict[str, Tensor]):
        msg = f"Invalid argument type {type(tensors)}. (expected dict[str, Tensor] but found {get_fullname(type(tensors))})"
        raise TypeError(msg)

    fpath = _setup_output_fpath(fpath, overwrite, make_parents)
    content = save(tensors, metadata)
    if fpath is not None:
        fpath.write_bytes(content)
    return content
