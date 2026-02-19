import os
from pathlib import Path
from typing import BinaryIO, TypedDict

from _typeshed import Incomplete
from pythonwrench.importlib import Placeholder
from torch import Tensor as Tensor
from torchaudio import _AudioMetaData
from torchaudio.io import CodecConfig as CodecConfig

msg: Incomplete

class CodecConfig(Placeholder): ...
class _AudioMetaData(Placeholder): ...

class AudioMetaDataDict(TypedDict):
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int
    encoding: str

def audio_metadata_to_dict(meta: _AudioMetaData) -> AudioMetaDataDict: ...
def dump_with_torchaudio(
    src: Tensor,
    uri: BinaryIO | str | Path | os.PathLike | None,
    sample_rate: int,
    channels_first: bool = True,
    format: str | None = None,
    encoding: str | None = None,
    bits_per_sample: int | None = None,
    buffer_size: int = 4096,
    backend: str | None = None,
    compression: CodecConfig | float | int | None = None,
    *,
    overwrite: bool = True,
    make_parents: bool = True,
) -> bytes: ...
def load_with_torchaudio(
    uri: BinaryIO | str | os.PathLike | Path,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    format: str | None = None,
    buffer_size: int = 4096,
    backend: str | None = None,
) -> tuple[Tensor, int]: ...
def dump_audio(*args, **kwargs) -> None: ...
def load_audio(*args, **kwargs) -> None: ...
