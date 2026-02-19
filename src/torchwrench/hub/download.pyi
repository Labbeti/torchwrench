from pathlib import Path

from _typeshed import Incomplete
from pythonwrench.hashlib import hash_file as hash_file
from pythonwrench.os import safe_rmdir as safe_rmdir

pylog: Incomplete

def download_file(
    url: str,
    dst: str | Path | None = ".",
    *,
    hash_prefix: str | None = None,
    make_parents: bool = False,
    make_intermediate: bool | None = None,
    verbose: int = 0,
) -> Path: ...
