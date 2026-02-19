from typing import overload

from torch import Tensor as Tensor

from torchwrench.core.make import DeviceLike as DeviceLike
from torchwrench.core.make import DTypeLike as DTypeLike
from torchwrench.core.make import as_device as as_device
from torchwrench.core.make import as_dtype as as_dtype
from torchwrench.nn.functional.multiclass import probs_to_onehot as probs_to_onehot
from torchwrench.types.tensor_subclasses import Tensor2D as Tensor2D
from torchwrench.types.tensor_subclasses import Tensor3D as Tensor3D

@overload
def multilabel_to_powerset(
    multilabel: Tensor, *, num_classes: int, max_set_size: int
) -> Tensor3D: ...
@overload
def multilabel_to_powerset(multilabel: Tensor, *, mapping: Tensor) -> Tensor3D: ...
@overload
def powerset_to_multilabel(
    powerset: Tensor, soft: bool = False, *, num_classes: int, max_set_size: int
) -> Tensor3D: ...
@overload
def powerset_to_multilabel(
    powerset: Tensor, soft: bool = False, *, mapping: Tensor
) -> Tensor3D: ...
def build_powerset_mapping(
    num_classes: int,
    max_set_size: int,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor2D: ...
def get_num_powerset_classes(num_classes: int, max_set_size: int) -> int: ...
