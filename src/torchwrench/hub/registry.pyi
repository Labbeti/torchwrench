from pathlib import Path
from typing import Any, Generic, Hashable, Mapping, TypedDict, TypeVar

from _typeshed import Incomplete
from pythonwrench.hashlib import HashName as HashName
from torch import Tensor as Tensor
from typing_extensions import NotRequired

from torchwrench.core.make import DeviceLike as DeviceLike
from torchwrench.core.make import as_device as as_device
from torchwrench.serialization.json import dump_json as dump_json
from torchwrench.serialization.json import load_json as load_json
from torchwrench.serialization.load_fn import LOAD_FNS as LOAD_FNS
from torchwrench.serialization.load_fn import LoadFnLike as LoadFnLike
from torchwrench.serialization.load_fn import load_torch as load_torch

from .paths import get_cache_dir as get_cache_dir

T_Hashable = TypeVar("T_Hashable", bound=Hashable)
pylog: Incomplete

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
        register_root: str | Path = "~/.cache/torch/hub/checkpoints",
    ) -> None: ...
    @property
    def infos(self) -> dict[T_Hashable, RegistryEntry]: ...
    @property
    def register_root(self) -> Path: ...
    @property
    def names(self) -> list[T_Hashable]: ...
    @property
    def paths(self) -> list[Path]: ...
    def get_path(self, name: T_Hashable) -> Path: ...
    def load_state_dict(
        self,
        name_or_path: T_Hashable | str | Path,
        *,
        device: DeviceLike = None,
        offline: bool = False,
        load_fn: LoadFnLike = ...,
        load_kwds: dict[str, Any] | None = None,
        verbose: int = 0,
    ) -> dict[str, Tensor]: ...
    def download_file(
        self,
        name: T_Hashable,
        force: bool = False,
        check_hash: bool = True,
        verbose: int = 0,
    ) -> tuple[Path, bool]: ...
    def remove_file(self, name: T_Hashable) -> None: ...
    def is_valid_hash(self, name: T_Hashable) -> bool: ...
    def save(self, path: str | Path) -> None: ...
    @classmethod
    def from_file(cls, path: str | Path) -> RegistryHub: ...
