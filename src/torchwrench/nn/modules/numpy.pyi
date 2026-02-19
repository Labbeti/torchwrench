from _typeshed import Incomplete
from torch import Tensor as Tensor

from torchwrench.core.make import DeviceLike as DeviceLike
from torchwrench.core.make import DTypeLike as DTypeLike
from torchwrench.core.packaging import NUMPY_AVAILABLE as NUMPY_AVAILABLE
from torchwrench.extras.numpy.definitions import np as np
from torchwrench.extras.numpy.functional import (
    ndarray_to_tensor as ndarray_to_tensor,
)
from torchwrench.extras.numpy.functional import (
    tensor_to_ndarray as tensor_to_ndarray,
)
from torchwrench.extras.numpy.functional import (
    to_ndarray as to_ndarray,
)
from torchwrench.nn.modules.module import Module as Module

class ToNDArray(Module):
    dtype: Incomplete
    force: Incomplete
    def __init__(
        self, *, dtype: str | np.dtype | None = None, force: bool = False
    ) -> None: ...
    def forward(self, x: Tensor | np.ndarray | list) -> np.ndarray: ...

class TensorToNDArray(Module):
    dtype: Incomplete
    force: Incomplete
    def __init__(
        self, *, dtype: str | np.dtype | None = None, force: bool = False
    ) -> None: ...
    def forward(self, x: Tensor) -> np.ndarray: ...

class NDArrayToTensor(Module):
    device: Incomplete
    dtype: Incomplete
    def __init__(
        self, *, device: DeviceLike = None, dtype: DTypeLike = None
    ) -> None: ...
    def forward(self, x: np.ndarray) -> Tensor: ...
