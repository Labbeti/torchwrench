import io
from argparse import Namespace
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from _typeshed import Incomplete
from pythonwrench.typing import DataclassInstance as DataclassInstance
from pythonwrench.typing import NamedTupleInstance as NamedTupleInstance
from typing_extensions import TypeAlias
from yaml import BaseLoader as BaseLoader
from yaml import FullLoader as FullLoader
from yaml import Loader as Loader
from yaml import MappingNode as MappingNode
from yaml import Node as Node
from yaml import SafeLoader as SafeLoader
from yaml import ScalarNode as ScalarNode
from yaml import SequenceNode as SequenceNode
from yaml import UnsafeLoader as UnsafeLoader
from yaml.parser import ParserError as ParserError
from yaml.scanner import ScannerError as ScannerError

from torchwrench.core.packaging import OMEGACONF_AVAILABLE as OMEGACONF_AVAILABLE
from torchwrench.core.packaging import YAML_AVAILABLE as YAML_AVAILABLE
from torchwrench.extras.yaml._yaml_fallback import CBaseLoader as CBaseLoader
from torchwrench.extras.yaml._yaml_fallback import CFullLoader as CFullLoader
from torchwrench.extras.yaml._yaml_fallback import CLoader as CLoader
from torchwrench.extras.yaml._yaml_fallback import CSafeLoader as CSafeLoader
from torchwrench.extras.yaml._yaml_fallback import CUnsafeLoader as CUnsafeLoader
from torchwrench.serialization.common import as_builtin as as_builtin

YamlLoaders: TypeAlias = (
    type[Loader]
    | type[BaseLoader]
    | type[FullLoader]
    | type[SafeLoader]
    | type[UnsafeLoader]
    | type[CLoader]
    | type[CBaseLoader]
    | type[CFullLoader]
    | type[CSafeLoader]
    | type[CUnsafeLoader]
)

def dump_yaml(
    data: Iterable[Any]
    | Mapping[str, Any]
    | Namespace
    | DataclassInstance
    | NamedTupleInstance,
    fpath: str | Path | None = None,
    *,
    overwrite: bool = True,
    to_builtins: bool = False,
    make_parents: bool = True,
    resolve: bool = False,
    encoding: str | None = "utf-8",
    sort_keys: bool = False,
    indent: int | None = None,
    width: int | None = 1000,
    allow_unicode: bool = True,
    **yaml_dump_kwds,
) -> str: ...
def dumps_yaml(*args, **kwargs) -> None: ...
def save_yaml(*args, **kwargs) -> None: ...
def load_yaml(
    file: str | Path | io.TextIOBase,
    *,
    Loader: YamlLoaders = ...,
    on_error: Literal["raise", "ignore"] = "raise",
) -> Any: ...
def loads_yaml(
    content: str | io.TextIOBase,
    *,
    Loader: YamlLoaders = ...,
    on_error: Literal["raise", "ignore"] = "raise",
) -> Any: ...
def read_yaml(*args, **kwargs) -> None: ...

class IgnoreTagLoader(SafeLoader):
    def construct_with_tag(self, tag: str, node: Node) -> Any: ...

class SplitTagLoader(SafeLoader):
    tag_key: Incomplete
    args_key: Incomplete
    def __init__(
        self, stream, *, tag_key: str = "_target_", args_key: str = "_args_"
    ) -> None: ...
    def construct_with_tag(self, tag: str, node: Node) -> Any: ...
