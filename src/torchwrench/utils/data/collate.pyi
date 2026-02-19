from typing import Any, TypeVar

from _typeshed import Incomplete
from pythonwrench.collections.collections import KeyMode as KeyMode
from pythonwrench.re import PatternListLike as PatternListLike

from torchwrench.nn.functional.padding import pad_and_stack_rec as pad_and_stack_rec
from torchwrench.nn.functional.predicate import (
    is_convertible_to_tensor as is_convertible_to_tensor,
)
from torchwrench.nn.functional.predicate import is_stackable as is_stackable

K = TypeVar("K")
V = TypeVar("V")

class CollateDict:
    key_mode: KeyMode
    def __init__(self, key_mode: KeyMode = "same") -> None: ...
    def __call__(self, batch_lst: list[dict[K, V]]) -> dict[K, list[V]]: ...

class AdvancedCollateDict:
    pad_values: Incomplete
    include_keys: Incomplete
    exclude_keys: Incomplete
    key_mode: KeyMode
    def __init__(
        self,
        pad_values: dict[str, Any] | None = None,
        include_keys: PatternListLike | None = None,
        exclude_keys: PatternListLike | None = None,
        key_mode: KeyMode = "same",
    ) -> None: ...
    def __call__(self, batch_lst: list[dict[str, Any]]) -> dict[str, Any]: ...
