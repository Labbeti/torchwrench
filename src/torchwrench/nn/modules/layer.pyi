from _typeshed import Incomplete
from torch import Tensor as Tensor

from torchwrench.core.make import DeviceLike as DeviceLike
from torchwrench.core.make import DTypeLike as DTypeLike
from torchwrench.core.make import as_device as as_device
from torchwrench.core.make import as_dtype as as_dtype
from torchwrench.types._typing import Tensor2D as Tensor2D

from .module import Module as Module

class PositionalEncoding(Module):
    dropout: Incomplete
    pos_embedding: Tensor
    def __init__(
        self,
        emb_size: int,
        dropout_p: float,
        maxlen: int = 5000,
        device: DeviceLike = None,
    ) -> None: ...
    def forward(self, token_emb: Tensor) -> Tensor: ...

def init_pos_emb(
    emb_size: int,
    maxlen: int = 5000,
    device: DeviceLike = None,
    dtype: DTypeLike = None,
) -> Tensor2D: ...
