from _typeshed import Incomplete
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer as Optimizer

pylog: Incomplete

class CosDecayScheduler(LambdaLR):
    def __init__(
        self, optimizer: Optimizer, n_steps: int, last_epoch: int = -1
    ) -> None: ...

class CosDecayRule:
    n_steps: Incomplete
    def __init__(self, n_steps: int) -> None: ...
    def __call__(self, step: int) -> float: ...
