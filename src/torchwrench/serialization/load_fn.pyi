import os
from pathlib import Path
from typing import Any, BinaryIO, Callable, TextIO, TypeVar, overload

from _typeshed import Incomplete
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
from torchwrench.extras.safetensors import load_safetensors as load_safetensors

from .common import SavingBackend as SavingBackend
from .csv import load_csv as load_csv
from .hdf import load_hdf as load_hdf
from .json import load_json as load_json
from .numpy import load_ndarray as load_ndarray
from .pickle import load_pickle as load_pickle
from .torch import load_torch as load_torch
from .torchaudio import load_with_torchaudio as load_with_torchaudio
from .yaml import load_yaml as load_yaml

T = TypeVar("T", covariant=True)
pylog: Incomplete
LoadFn: TypeAlias = Callable[[Any], T]
LoadFnLike: TypeAlias = LoadFn[T] | SavingBackend
LOAD_FNS: dict[SavingBackend, LoadFn[Any]]

@overload
def load(
    fpath: TextIO | BinaryIO, *args, saving_backend: SavingBackend = "torch", **kwargs
) -> Any: ...
@overload
def load(
    fpath: str | Path | os.PathLike,
    *args,
    saving_backend: SavingBackend | None = "torch",
    **kwargs,
) -> Any: ...
def read(*args, **kwargs) -> None: ...
