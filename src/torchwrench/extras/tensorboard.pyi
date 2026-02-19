from pathlib import Path
from typing import Iterable, TypedDict

from _typeshed import Incomplete
from typing_extensions import NotRequired

msg: str
pylog: Incomplete

class TensorboardEvent(TypedDict):
    wall_time: float
    step: int
    tag: str
    dtype: str
    value: str | float
    string_val: NotRequired[str]
    float_val: NotRequired[list[float]]

def load_tfevents(
    fpath: str | Path,
    *,
    cast_float_and_str: bool = True,
    ignore_underscore_tags: bool = True,
    verbose: int = 0,
) -> list[TensorboardEvent]: ...
def load_tfevents_files(
    paths_or_patterns: str | Path | Iterable[str | Path],
    *,
    cast_float_and_str: bool = True,
    ignore_underscore_tags: bool = True,
    verbose: int = 0,
) -> dict[str, list[TensorboardEvent]]: ...
def get_tfevents_duration(fpath: str | Path, verbose: int = 0) -> float: ...
