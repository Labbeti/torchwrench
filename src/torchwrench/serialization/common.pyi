from typing import TypeVar

from _typeshed import Incomplete

from torchwrench.core.packaging import (
    H5PY_AVAILABLE as H5PY_AVAILABLE,
)
from torchwrench.core.packaging import (
    NUMPY_AVAILABLE as NUMPY_AVAILABLE,
)
from torchwrench.core.packaging import (
    OMEGACONF_AVAILABLE as OMEGACONF_AVAILABLE,
)
from torchwrench.core.packaging import (
    PANDAS_AVAILABLE as PANDAS_AVAILABLE,
)
from torchwrench.core.packaging import (
    SAFETENSORS_AVAILABLE as SAFETENSORS_AVAILABLE,
)
from torchwrench.core.packaging import (
    TORCHAUDIO_AVAILABLE as TORCHAUDIO_AVAILABLE,
)
from torchwrench.core.packaging import (
    YAML_AVAILABLE as YAML_AVAILABLE,
)

T = TypeVar("T")
pylog: Incomplete
SavingBackend: Incomplete
PATTERN_TO_BACKEND: dict[str, SavingBackend]
BACKEND_TO_PATTERN: dict[SavingBackend, str]
