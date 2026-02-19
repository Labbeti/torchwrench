from io import TextIOBase
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, overload

from _typeshed import Incomplete

from torchwrench.core.packaging import PANDAS_AVAILABLE as PANDAS_AVAILABLE
from torchwrench.extras.pandas import pd as pd
from torchwrench.serialization.common import as_builtin as as_builtin

OrientExtended: Incomplete
CSVBackend: Incomplete

def dump_csv(
    data: Iterable[Mapping[str, Any]] | Mapping[str, Iterable[Any]] | Iterable,
    fpath: str | Path | None = None,
    *,
    overwrite: bool = True,
    to_builtins: bool = False,
    make_parents: bool = True,
    backend: CSVBackend = "auto",
    header: bool | Literal["auto"] = "auto",
    **csv_backend_kwds,
) -> str: ...
def dumps_csv(*args, **kwargs) -> None: ...
def save_csv(*args, **kwargs) -> None: ...
@overload
def load_csv(
    fpath: str | Path | TextIOBase,
    /,
    *,
    orient: Literal["list"] = "list",
    header: bool = True,
    comment_start: str | None = None,
    strip_content: bool = False,
    backend: CSVBackend = "auto",
    delimiter: str | None = None,
    **csv_backend_kwds,
) -> list[dict[str, Any]]: ...
@overload
def load_csv(
    fpath: str | Path | TextIOBase,
    /,
    *,
    orient: Literal["dict"],
    header: bool = True,
    comment_start: str | None = None,
    strip_content: bool = False,
    backend: CSVBackend = "auto",
    delimiter: str | None = None,
    **csv_backend_kwds,
) -> dict[str, list[Any]]: ...
@overload
def load_csv(
    fpath: str | Path | TextIOBase,
    /,
    *,
    orient: Literal["dataframe"],
    header: bool = True,
    comment_start: str | None = None,
    strip_content: bool = False,
    backend: CSVBackend = "auto",
    delimiter: str | None = None,
    **csv_backend_kwds,
) -> pd.DataFrame: ...
def loads_csv(*args, **kwargs) -> None: ...
def read_csv(*args, **kwargs) -> None: ...
