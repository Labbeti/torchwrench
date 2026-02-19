from torch import Tensor

from torchwrench.core.make import (
    DeviceLike as DeviceLike,
)
from torchwrench.core.make import (
    DTypeLike as DTypeLike,
)
from torchwrench.core.make import (
    GeneratorLike as GeneratorLike,
)
from torchwrench.core.make import (
    as_device as as_device,
)
from torchwrench.core.make import (
    as_dtype as as_dtype,
)
from torchwrench.core.make import (
    as_generator as as_generator,
)
from torchwrench.types import (
    BuiltinNumber as BuiltinNumber,
)
from torchwrench.types import (
    LongTensor as LongTensor,
)
from torchwrench.types import (
    LongTensor1D as LongTensor1D,
)
from torchwrench.types import (
    Tensor1D as Tensor1D,
)
from torchwrench.types import (
    is_builtin_number as is_builtin_number,
)

def get_inverse_perm(indices: Tensor, dim: int = -1) -> Tensor: ...
def randperm_diff(
    size: int,
    generator: GeneratorLike = None,
    device: DeviceLike = None,
    *,
    dtype: DTypeLike = ...,
) -> LongTensor1D: ...
def get_perm_indices(x1: Tensor, x2: Tensor) -> LongTensor: ...
def insert_at_indices(
    x: Tensor, indices: Tensor | list | BuiltinNumber, values: BuiltinNumber | Tensor
) -> Tensor1D: ...
def remove_at_indices(
    x: Tensor, indices: Tensor | list | BuiltinNumber
) -> Tensor1D: ...
