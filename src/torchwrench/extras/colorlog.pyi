from _typeshed import Incomplete
from colorlog import ColoredFormatter

msg: str
pylog: Incomplete
LOG_COLORS: Incomplete

def get_colored_formatter(slurm_rank: bool = False) -> ColoredFormatter: ...
def get_colored_format(slurm_rank: bool = False) -> str: ...
