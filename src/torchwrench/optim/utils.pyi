from typing import Iterable

from _typeshed import Incomplete
from torch import nn
from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer as Optimizer

pylog: Incomplete

def get_lr(optim: Optimizer, idx: int = 0, key: str = "lr") -> float: ...
def get_lrs(optim: Optimizer, key: str = "lr") -> list[float]: ...
def create_params_groups_bias(
    model: nn.Module | Iterable[tuple[str, Parameter]],
    weight_decay: float,
    skip_list: Iterable[str] | None = (),
    verbose: int = 2,
) -> list[dict[str, list[Parameter] | float]]: ...
