from typing import Iterable

from _typeshed import Incomplete
from torch import Tensor as Tensor

from torchwrench.nn.functional.mask import masked_mean as masked_mean
from torchwrench.nn.functional.mask import masked_sum as masked_sum

from .module import Module as Module

class MaskedMean(Module):
    dim: Incomplete
    def __init__(self, dim: None | int | Iterable[int] = None) -> None: ...
    def forward(self, tensor: Tensor, non_pad_mask: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class MaskedSum(Module):
    dim: Incomplete
    def __init__(self, dim: None | int | Iterable[int] = None) -> None: ...
    def forward(self, tensor: Tensor, non_pad_mask: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...
