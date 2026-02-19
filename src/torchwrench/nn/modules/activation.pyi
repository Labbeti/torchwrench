from typing import Iterable

from _typeshed import Incomplete
from torch import Tensor as Tensor

from torchwrench.nn.functional.activation import (
    log_softmax_multidim as log_softmax_multidim,
)
from torchwrench.nn.functional.activation import softmax_multidim as softmax_multidim

from .module import Module as Module

class SoftmaxMultidim(Module):
    dims: Incomplete
    def __init__(self, dims: Iterable[int] | None = (-1,)) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class LogSoftmaxMultidim(Module):
    dims: Incomplete
    def __init__(self, dims: Iterable[int] | None = (-1,)) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...
