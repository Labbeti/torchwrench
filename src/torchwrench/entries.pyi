from torchwrench.hub.paths import (
    get_cache_dir as get_cache_dir,
)
from torchwrench.hub.paths import (
    get_dir as get_dir,
)
from torchwrench.hub.paths import (
    get_tmp_dir as get_tmp_dir,
)
from torchwrench.utils.data.dataloader import (
    get_auto_num_cpus as get_auto_num_cpus,
)
from torchwrench.utils.data.dataloader import (
    get_auto_num_gpus as get_auto_num_gpus,
)

def get_package_repository_path() -> str: ...
def get_install_info() -> dict[str, str | int]: ...
def print_install_info() -> None: ...
