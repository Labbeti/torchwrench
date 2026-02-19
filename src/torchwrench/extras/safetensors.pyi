from pathlib import Path
from typing import Any, Literal, overload

from torch import Tensor

@overload
def load_safetensors(
    fpath: str | Path, *, device: str = "cpu", return_metadata: Literal[False] = False
) -> dict[str, Tensor]: ...
@overload
def load_safetensors(
    fpath: str | Path, *, device: str = "cpu", return_metadata: Literal[True]
) -> tuple[dict[str, Tensor], dict[str, str]]: ...
@overload
def dump_safetensors(
    tensors: dict[str, Tensor],
    fpath: str | Path | None = None,
    metadata: dict[str, str] | None = None,
    *,
    overwrite: bool = True,
    make_parents: bool = True,
    convert_to_tensor: Literal[False] = False,
) -> bytes: ...
@overload
def dump_safetensors(
    tensors: dict[str, Any],
    fpath: str | Path | None = None,
    metadata: dict[str, str] | None = None,
    *,
    overwrite: bool = True,
    make_parents: bool = True,
    convert_to_tensor: Literal[True],
) -> bytes: ...
