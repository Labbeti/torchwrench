from typing import Iterable

from torch import Tensor as Tensor

def softmax_multidim(x: Tensor, *, dims: Iterable[int] | None = (-1,)) -> Tensor: ...
def log_softmax_multidim(
    x: Tensor, *, dims: Iterable[int] | None = (-1,)
) -> Tensor: ...
