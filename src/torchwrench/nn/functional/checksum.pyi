import torch
from pythonwrench.checksum import checksum_any as checksum_any
from torch import Tensor, nn

from torchwrench.extras.numpy import np as np
from torchwrench.extras.pandas import pd as pd
from torchwrench.nn.functional.predicate import (
    is_complex as is_complex,
)
from torchwrench.nn.functional.predicate import (
    is_floating_point as is_floating_point,
)

def checksum_dataframe(x: pd.DataFrame, **kwargs) -> int: ...
def checksum_series(x: pd.Series, **kwargs) -> int: ...
def checksum_dtype(x: torch.dtype | np.dtype, **kwargs) -> int: ...
def checksum_module(
    x: nn.Module,
    *,
    only_trainable: bool = False,
    with_names: bool = False,
    buffers: bool = False,
    training: bool = False,
    **kwargs,
) -> int: ...
def checksum_tensor(x: Tensor, **kwargs) -> int: ...
def checksum_numpy(x: np.ndarray | np.generic, **kwargs) -> int: ...
