import os
from pathlib import Path
from typing import Any, BinaryIO, Callable, overload

from typing_extensions import TypeAlias

from torchwrench.core.packaging import (
    H5PY_AVAILABLE as H5PY_AVAILABLE,
)
from torchwrench.core.packaging import (
    NUMPY_AVAILABLE as NUMPY_AVAILABLE,
)
from torchwrench.core.packaging import (
    SAFETENSORS_AVAILABLE as SAFETENSORS_AVAILABLE,
)
from torchwrench.core.packaging import (
    TORCHAUDIO_AVAILABLE as TORCHAUDIO_AVAILABLE,
)
from torchwrench.core.packaging import (
    YAML_AVAILABLE as YAML_AVAILABLE,
)
from torchwrench.extras.safetensors import dump_safetensors as dump_safetensors

from .common import SavingBackend as SavingBackend
from .csv import dump_csv as dump_csv
from .hdf import dump_hdf as dump_hdf
from .json import dump_json as dump_json
from .numpy import dump_ndarray as dump_ndarray
from .pickle import dump_pickle as dump_pickle
from .torch import dump_torch as dump_torch
from .torchaudio import dump_with_torchaudio as dump_with_torchaudio
from .yaml import dump_yaml as dump_yaml

DumpFn: TypeAlias = Callable[..., Any]
DumpFnLike: TypeAlias = DumpFn | SavingBackend
DUMP_FNS: dict[SavingBackend, DumpFn]

@overload
def dump(
    obj: Any,
    fpath: None | BinaryIO = None,
    *args,
    saving_backend: SavingBackend = "torch",
    **kwargs,
) -> str | bytes: ...
@overload
def dump(
    obj: Any,
    fpath: str | Path | os.PathLike,
    *args,
    saving_backend: SavingBackend | None = "torch",
    **kwargs,
) -> str | bytes: ...
def save(*args, **kwargs) -> None: ...
