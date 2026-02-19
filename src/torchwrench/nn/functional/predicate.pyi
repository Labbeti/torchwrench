from typing import Any, Iterable, TypeVar, overload

from torch import Tensor
from typing_extensions import TypeGuard

from torchwrench.extras.numpy import (
    ACCEPTED_NUMPY_DTYPES as ACCEPTED_NUMPY_DTYPES,
)
from torchwrench.extras.numpy import (
    np as np,
)
from torchwrench.extras.numpy import (
    numpy_all_eq as numpy_all_eq,
)
from torchwrench.extras.numpy import (
    numpy_all_ne as numpy_all_ne,
)
from torchwrench.extras.numpy import (
    numpy_is_complex as numpy_is_complex,
)
from torchwrench.extras.numpy import (
    numpy_is_floating_point as numpy_is_floating_point,
)
from torchwrench.nn.functional.others import nelement as nelement
from torchwrench.types._typing import (
    ComplexFloatingTensor as ComplexFloatingTensor,
)
from torchwrench.types._typing import (
    FloatingTensor as FloatingTensor,
)
from torchwrench.types._typing import (
    ScalarLike as ScalarLike,
)
from torchwrench.types._typing import (
    T_TensorOrArray as T_TensorOrArray,
)
from torchwrench.types._typing import (
    Tensor0D as Tensor0D,
)
from torchwrench.types._typing import (
    TensorOrArray as TensorOrArray,
)
from torchwrench.types.guards import is_scalar_like as is_scalar_like

T = TypeVar("T")
U = TypeVar("U")

def is_stackable(
    tensors: list[Any] | tuple[Any, ...],
) -> TypeGuard[list[Tensor] | tuple[Tensor, ...]]: ...
def is_convertible_to_tensor(x: Any) -> bool: ...
@overload
def is_floating_point(x: Tensor) -> TypeGuard[FloatingTensor]: ...
@overload
def is_floating_point(x: np.ndarray) -> TypeGuard[np.ndarray]: ...
@overload
def is_floating_point(x: float) -> TypeGuard[float]: ...
@overload
def is_floating_point(x: Any) -> TypeGuard[FloatingTensor | np.ndarray | float]: ...
@overload
def is_complex(x: Tensor) -> TypeGuard[ComplexFloatingTensor]: ...
@overload
def is_complex(x: np.ndarray) -> TypeGuard[np.ndarray]: ...
@overload
def is_complex(x: complex) -> TypeGuard[complex]: ...
@overload
def is_complex(x: Any) -> TypeGuard[ComplexFloatingTensor | np.ndarray | complex]: ...
def is_sorted(
    x: Tensor | np.ndarray | Iterable, *, reverse: bool = False, strict: bool = False
) -> bool: ...
@overload
def all_eq(
    x: Tensor | np.ndarray | ScalarLike | Iterable, dim: None = None
) -> bool: ...
@overload
def all_eq(x: T_TensorOrArray, dim: int) -> T_TensorOrArray: ...
def all_ne(x: Tensor | np.ndarray | ScalarLike | Iterable) -> bool: ...
def is_full(x: TensorOrArray, target: Any = ...) -> bool: ...
def is_unique(*args, **kwargs) -> None: ...
