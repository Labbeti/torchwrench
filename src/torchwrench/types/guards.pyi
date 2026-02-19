from typing import Any

from typing_extensions import TypeGuard, TypeIs

from torchwrench.core.make import DTypeLike as DTypeLike
from torchwrench.core.make import as_dtype as as_dtype
from torchwrench.extras.numpy import (
    is_numpy_number_like as is_numpy_number_like,
)
from torchwrench.extras.numpy import (
    is_numpy_scalar_like as is_numpy_scalar_like,
)
from torchwrench.extras.numpy import (
    np as np,
)

from ._typing import (
    IntegralTensor as IntegralTensor,
)
from ._typing import (
    NumberLike as NumberLike,
)
from ._typing import (
    ScalarLike as ScalarLike,
)
from ._typing import (
    Tensor0D as Tensor0D,
)
from ._typing import (
    TensorOrArray as TensorOrArray,
)

def is_number_like(x: Any) -> TypeGuard[NumberLike]: ...
def is_scalar_like(x: Any) -> TypeGuard[ScalarLike]: ...
def is_tensor_or_array(x: Any) -> TypeIs[TensorOrArray]: ...
def is_integral_dtype(dtype: DTypeLike) -> bool: ...
