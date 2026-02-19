import os
from typing import IO, Any, BinaryIO

from typing_extensions import TypeAlias

FileLike: TypeAlias = str | os.PathLike | BinaryIO | IO[bytes]
MapLocationLike: TypeAlias

def dump_torch(
    obj: object,
    f: FileLike | None = None,
    pickle_module: Any = ...,
    pickle_protocol: int = ...,
    _use_new_zipfile_serialization: bool = True,
    _disable_byteorder_record: bool = False,
    *,
    overwrite: bool = True,
    make_parents: bool = True,
) -> bytes: ...
def load_torch(
    f: FileLike,
    map_location: MapLocationLike = None,
    pickle_module: Any = None,
    *,
    weights_only: bool = ...,
    mmap: bool | None = None,
    **pickle_load_args: Any,
) -> Any: ...
