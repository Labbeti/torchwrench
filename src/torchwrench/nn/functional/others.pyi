from typing import Any, Callable, Iterable, Literal, TypeVar, overload

from pythonwrench.typing import BuiltinNumber as BuiltinNumber
from pythonwrench.typing import T_BuiltinNumber as T_BuiltinNumber
from torch import Tensor
from torch import nn as nn

from torchwrench.extras.numpy import np as np
from torchwrench.extras.pandas import pd as pd
from torchwrench.types._typing import LongTensor as LongTensor
from torchwrench.types._typing import ScalarLike as ScalarLike
from torchwrench.types._typing import T_TensorOrArray as T_TensorOrArray
from torchwrench.types._typing import TensorOrArray as TensorOrArray
from torchwrench.types.guards import is_scalar_like as is_scalar_like
from torchwrench.types.tensor_subclasses import Tensor0D as Tensor0D
from torchwrench.types.tensor_subclasses import Tensor1D as Tensor1D
from torchwrench.types.tensor_subclasses import Tensor2D as Tensor2D
from torchwrench.types.tensor_subclasses import Tensor3D as Tensor3D
from torchwrench.utils import return_types as return_types

T = TypeVar("T", covariant=True)
U = TypeVar("U", covariant=True)

def count_parameters(
    model: nn.Module,
    *,
    recurse: bool = True,
    only_trainable: bool = False,
    buffers: bool = False,
) -> int: ...
def find(
    value: Any,
    x: Tensor,
    *,
    default: None | Tensor | BuiltinNumber = None,
    dim: int = -1,
) -> LongTensor: ...
@overload
def get_ndim(
    x: ScalarLike | Tensor | np.ndarray | Iterable,
    *,
    use_first_for_list_tuple: bool = False,
    return_indicator: Literal[False] = False,
    return_default_on_invalid: Literal[False] = False,
    default: Any = -1,
    return_valid: bool | None = None,
) -> int: ...
@overload
def get_ndim(
    x: ScalarLike | Tensor | np.ndarray | Iterable,
    *,
    use_first_for_list_tuple: bool = False,
    return_indicator: Literal[False] = False,
    return_default_on_invalid: bool,
    default: U = -1,
    return_valid: bool | None = None,
) -> int | U: ...
@overload
def get_ndim(
    x: ScalarLike | Tensor | np.ndarray | Iterable,
    *,
    use_first_for_list_tuple: bool = False,
    return_indicator: Literal[True],
    return_default_on_invalid: Literal[False] = False,
    default: Any = -1,
    return_valid: bool | None = None,
) -> return_types.ndim[int]: ...
@overload
def get_ndim(
    x: ScalarLike | Tensor | np.ndarray | Iterable,
    *,
    use_first_for_list_tuple: bool = False,
    return_indicator: Literal[True],
    return_default_on_invalid: bool,
    default: U = -1,
    return_valid: bool | None = None,
) -> return_types.ndim[int | U]: ...
def ndim(*args, **kwargs) -> None: ...
@overload
def get_shape(
    x: ScalarLike
    | Tensor
    | np.ndarray
    | pd.DataFrame
    | list
    | tuple
    | set
    | frozenset
    | dict,
    *,
    output_type: Callable[[tuple[int, ...]], T] = ...,
    use_first_for_list_tuple: bool = False,
    return_indicator: Literal[False] = False,
    return_default_on_invalid: Literal[False] = False,
    default: Any = (),
    return_valid: bool | None = None,
) -> T: ...
@overload
def get_shape(
    x: ScalarLike
    | Tensor
    | np.ndarray
    | pd.DataFrame
    | list
    | tuple
    | set
    | frozenset
    | dict,
    *,
    output_type: Callable[[tuple[int, ...]], T] = ...,
    use_first_for_list_tuple: bool = False,
    return_indicator: Literal[False] = False,
    return_default_on_invalid: bool,
    default: U = (),
    return_valid: bool | None = None,
) -> T | U: ...
@overload
def get_shape(
    x: ScalarLike
    | Tensor
    | np.ndarray
    | pd.DataFrame
    | list
    | tuple
    | set
    | frozenset
    | dict,
    *,
    output_type: Callable[[tuple[int, ...]], T] = ...,
    use_first_for_list_tuple: bool = False,
    return_indicator: Literal[True],
    return_default_on_invalid: Literal[False] = False,
    default: Any = (),
    return_valid: bool | None = None,
) -> return_types.shape[T]: ...
@overload
def get_shape(
    x: ScalarLike
    | Tensor
    | np.ndarray
    | pd.DataFrame
    | list
    | tuple
    | set
    | frozenset
    | dict,
    *,
    output_type: Callable[[tuple[int, ...]], T] = ...,
    use_first_for_list_tuple: bool = False,
    return_indicator: Literal[True],
    return_default_on_invalid: bool,
    default: U = (),
    return_valid: bool | None = None,
) -> return_types.shape[T | U]: ...
def shape(*args, **kwargs) -> None: ...
def ranks(x: Tensor, dim: int = -1, descending: bool = False) -> LongTensor: ...
def nelement(x: ScalarLike | Tensor | np.ndarray | Iterable) -> int: ...
@overload
def prod(
    x: T_TensorOrArray, *, dim: int | None = None, start: Any = 1
) -> T_TensorOrArray: ...
@overload
def prod(
    x: Iterable[T_BuiltinNumber], *, dim: Any = None, start: T_BuiltinNumber = 1
) -> T_BuiltinNumber: ...
def average_power(
    x: T_TensorOrArray, dim: int | tuple[int, ...] | None = -1
) -> T_TensorOrArray: ...
def mse(
    x1: Tensor, x2: Tensor, *, dim: int | tuple[int, ...] | None = None
) -> Tensor: ...
def rmse(
    x1: Tensor, x2: Tensor, *, dim: int | tuple[int, ...] | None = None
) -> Tensor: ...
def deep_equal(x: T, y: T, *args: T) -> bool: ...
@overload
def stack(
    tensors: list[Tensor0D] | tuple[Tensor0D, ...],
    dim: int = 0,
    *,
    out: Tensor1D | None = None,
) -> Tensor1D: ...
@overload
def stack(
    tensors: list[Tensor1D] | tuple[Tensor1D, ...],
    dim: int = 0,
    *,
    out: Tensor2D | None = None,
) -> Tensor2D: ...
@overload
def stack(
    tensors: list[Tensor2D] | tuple[Tensor2D, ...],
    dim: int = 0,
    *,
    out: Tensor3D | None = None,
) -> Tensor3D: ...
@overload
def stack(
    tensors: list[Tensor] | tuple[Tensor, ...],
    dim: int = 0,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def cat(
    tensors: list[Tensor] | tuple[Tensor, ...],
    dim: int = 0,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def concat(*args, **kwargs) -> None: ...
