from typing import Any

from pythonwrench.importlib import Placeholder

class DynamicItemDataset(Placeholder):
    def __len__(self) -> int: ...
    def __getitem__(self, *args, **kwargs) -> Any: ...
