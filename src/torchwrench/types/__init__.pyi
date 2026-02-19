from pythonwrench.typing import *
from torch.types import *

from torchwrench.core.dtype_enum import DTypeEnum as DTypeEnum
from torchwrench.core.make import (
    DeviceLike as DeviceLike,
)
from torchwrench.core.make import (
    DTypeLike as DTypeLike,
)
from torchwrench.core.make import (
    GeneratorLike as GeneratorLike,
)

from ._typing import (
    T_Tensor as T_Tensor,
)
from ._typing import (
    T_TensorOrArray as T_TensorOrArray,
)
from ._typing import (
    TensorOrArray as TensorOrArray,
)
from .guards import (
    is_builtin_number as is_builtin_number,
)
from .guards import (
    is_builtin_scalar as is_builtin_scalar,
)
from .guards import (
    is_integral_dtype as is_integral_dtype,
)
from .guards import (
    is_number_like as is_number_like,
)
from .guards import (
    is_numpy_number_like as is_numpy_number_like,
)
from .guards import (
    is_numpy_scalar_like as is_numpy_scalar_like,
)
from .guards import (
    is_scalar_like as is_scalar_like,
)
from .guards import (
    is_tensor_or_array as is_tensor_or_array,
)
from .tensor_subclasses import (
    BoolTensor as BoolTensor,
)
from .tensor_subclasses import (
    BoolTensor0D as BoolTensor0D,
)
from .tensor_subclasses import (
    BoolTensor1D as BoolTensor1D,
)
from .tensor_subclasses import (
    BoolTensor2D as BoolTensor2D,
)
from .tensor_subclasses import (
    BoolTensor3D as BoolTensor3D,
)
from .tensor_subclasses import (
    ByteTensor as ByteTensor,
)
from .tensor_subclasses import (
    ByteTensor0D as ByteTensor0D,
)
from .tensor_subclasses import (
    ByteTensor1D as ByteTensor1D,
)
from .tensor_subclasses import (
    ByteTensor2D as ByteTensor2D,
)
from .tensor_subclasses import (
    ByteTensor3D as ByteTensor3D,
)
from .tensor_subclasses import (
    CDoubleTensor as CDoubleTensor,
)
from .tensor_subclasses import (
    CDoubleTensor0D as CDoubleTensor0D,
)
from .tensor_subclasses import (
    CDoubleTensor1D as CDoubleTensor1D,
)
from .tensor_subclasses import (
    CDoubleTensor2D as CDoubleTensor2D,
)
from .tensor_subclasses import (
    CDoubleTensor3D as CDoubleTensor3D,
)
from .tensor_subclasses import (
    CFloatTensor as CFloatTensor,
)
from .tensor_subclasses import (
    CFloatTensor0D as CFloatTensor0D,
)
from .tensor_subclasses import (
    CFloatTensor1D as CFloatTensor1D,
)
from .tensor_subclasses import (
    CFloatTensor2D as CFloatTensor2D,
)
from .tensor_subclasses import (
    CFloatTensor3D as CFloatTensor3D,
)
from .tensor_subclasses import (
    CharTensor as CharTensor,
)
from .tensor_subclasses import (
    CharTensor0D as CharTensor0D,
)
from .tensor_subclasses import (
    CharTensor1D as CharTensor1D,
)
from .tensor_subclasses import (
    CharTensor2D as CharTensor2D,
)
from .tensor_subclasses import (
    CharTensor3D as CharTensor3D,
)
from .tensor_subclasses import (
    ComplexFloatingTensor as ComplexFloatingTensor,
)
from .tensor_subclasses import (
    ComplexFloatingTensor0D as ComplexFloatingTensor0D,
)
from .tensor_subclasses import (
    ComplexFloatingTensor1D as ComplexFloatingTensor1D,
)
from .tensor_subclasses import (
    ComplexFloatingTensor2D as ComplexFloatingTensor2D,
)
from .tensor_subclasses import (
    ComplexFloatingTensor3D as ComplexFloatingTensor3D,
)
from .tensor_subclasses import (
    DoubleTensor as DoubleTensor,
)
from .tensor_subclasses import (
    DoubleTensor0D as DoubleTensor0D,
)
from .tensor_subclasses import (
    DoubleTensor1D as DoubleTensor1D,
)
from .tensor_subclasses import (
    DoubleTensor2D as DoubleTensor2D,
)
from .tensor_subclasses import (
    DoubleTensor3D as DoubleTensor3D,
)
from .tensor_subclasses import (
    FloatingTensor as FloatingTensor,
)
from .tensor_subclasses import (
    FloatingTensor0D as FloatingTensor0D,
)
from .tensor_subclasses import (
    FloatingTensor1D as FloatingTensor1D,
)
from .tensor_subclasses import (
    FloatingTensor2D as FloatingTensor2D,
)
from .tensor_subclasses import (
    FloatingTensor3D as FloatingTensor3D,
)
from .tensor_subclasses import (
    FloatTensor as FloatTensor,
)
from .tensor_subclasses import (
    FloatTensor0D as FloatTensor0D,
)
from .tensor_subclasses import (
    FloatTensor1D as FloatTensor1D,
)
from .tensor_subclasses import (
    FloatTensor2D as FloatTensor2D,
)
from .tensor_subclasses import (
    FloatTensor3D as FloatTensor3D,
)
from .tensor_subclasses import (
    HalfTensor as HalfTensor,
)
from .tensor_subclasses import (
    HalfTensor0D as HalfTensor0D,
)
from .tensor_subclasses import (
    HalfTensor1D as HalfTensor1D,
)
from .tensor_subclasses import (
    HalfTensor2D as HalfTensor2D,
)
from .tensor_subclasses import (
    HalfTensor3D as HalfTensor3D,
)
from .tensor_subclasses import (
    IntegralTensor as IntegralTensor,
)
from .tensor_subclasses import (
    IntegralTensor0D as IntegralTensor0D,
)
from .tensor_subclasses import (
    IntegralTensor1D as IntegralTensor1D,
)
from .tensor_subclasses import (
    IntegralTensor2D as IntegralTensor2D,
)
from .tensor_subclasses import (
    IntegralTensor3D as IntegralTensor3D,
)
from .tensor_subclasses import (
    IntTensor as IntTensor,
)
from .tensor_subclasses import (
    IntTensor0D as IntTensor0D,
)
from .tensor_subclasses import (
    IntTensor1D as IntTensor1D,
)
from .tensor_subclasses import (
    IntTensor2D as IntTensor2D,
)
from .tensor_subclasses import (
    IntTensor3D as IntTensor3D,
)
from .tensor_subclasses import (
    LongTensor as LongTensor,
)
from .tensor_subclasses import (
    LongTensor0D as LongTensor0D,
)
from .tensor_subclasses import (
    LongTensor1D as LongTensor1D,
)
from .tensor_subclasses import (
    LongTensor2D as LongTensor2D,
)
from .tensor_subclasses import (
    LongTensor3D as LongTensor3D,
)
from .tensor_subclasses import (
    RealTensor as RealTensor,
)
from .tensor_subclasses import (
    RealTensor0D as RealTensor0D,
)
from .tensor_subclasses import (
    RealTensor1D as RealTensor1D,
)
from .tensor_subclasses import (
    RealTensor2D as RealTensor2D,
)
from .tensor_subclasses import (
    RealTensor3D as RealTensor3D,
)
from .tensor_subclasses import (
    ShortTensor as ShortTensor,
)
from .tensor_subclasses import (
    ShortTensor0D as ShortTensor0D,
)
from .tensor_subclasses import (
    ShortTensor1D as ShortTensor1D,
)
from .tensor_subclasses import (
    ShortTensor2D as ShortTensor2D,
)
from .tensor_subclasses import (
    ShortTensor3D as ShortTensor3D,
)
from .tensor_subclasses import (
    SignedIntegerTensor as SignedIntegerTensor,
)
from .tensor_subclasses import (
    SignedIntegerTensor0D as SignedIntegerTensor0D,
)
from .tensor_subclasses import (
    SignedIntegerTensor1D as SignedIntegerTensor1D,
)
from .tensor_subclasses import (
    SignedIntegerTensor2D as SignedIntegerTensor2D,
)
from .tensor_subclasses import (
    SignedIntegerTensor3D as SignedIntegerTensor3D,
)
from .tensor_subclasses import (
    Tensor as Tensor,
)
from .tensor_subclasses import (
    Tensor0D as Tensor0D,
)
from .tensor_subclasses import (
    Tensor1D as Tensor1D,
)
from .tensor_subclasses import (
    Tensor2D as Tensor2D,
)
from .tensor_subclasses import (
    Tensor3D as Tensor3D,
)
from .tensor_subclasses import (
    UnsignedIntegerTensor as UnsignedIntegerTensor,
)
from .tensor_subclasses import (
    UnsignedIntegerTensor0D as UnsignedIntegerTensor0D,
)
from .tensor_subclasses import (
    UnsignedIntegerTensor1D as UnsignedIntegerTensor1D,
)
from .tensor_subclasses import (
    UnsignedIntegerTensor2D as UnsignedIntegerTensor2D,
)
from .tensor_subclasses import (
    UnsignedIntegerTensor3D as UnsignedIntegerTensor3D,
)
