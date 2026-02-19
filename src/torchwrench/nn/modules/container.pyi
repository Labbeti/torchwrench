from typing import Callable, Generic, Iterable, Mapping, overload

from _typeshed import Incomplete
from torch import nn
from typing_extensions import Concatenate, ParamSpec

from ._mixins import ConfigModule as ConfigModule
from ._mixins import DeviceDetectMode as DeviceDetectMode
from ._mixins import EModule as EModule
from ._mixins import ESequential as ESequential
from ._mixins import InType as InType
from ._mixins import OutType as OutType
from ._mixins import OutType3 as OutType3
from ._mixins import TypedModuleLike as TypedModuleLike

P = ParamSpec("P")

class EModuleList(
    EModule[InType, list[OutType3]], nn.ModuleList, Generic[InType, OutType3]
):
    @overload
    def __init__(
        self,
        modules: Iterable[TypedModuleLike[InType, OutType3]] | None = None,
        *,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        modules: Iterable[nn.Module] | None = None,
        *,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = ...,
    ) -> None: ...
    def forward(self, *args: InType, **kwargs: InType) -> list[OutType3]: ...

class EModuleDict(
    EModule[InType, dict[str, OutType3]], nn.ModuleDict, Generic[InType, OutType3]
):
    @overload
    def __init__(
        self,
        modules: Mapping[str, TypedModuleLike[InType, OutType3]] | None = None,
        *,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        modules: Mapping[str, nn.Module] | None = None,
        *,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = ...,
    ) -> None: ...
    def forward(self, *args: InType, **kwargs: InType) -> dict[str, OutType3]: ...

class EModulePartial(EModule[InType, OutType], Generic[InType, OutType]):
    fn: Incomplete
    args: Incomplete
    kwargs: Incomplete
    def __init__(
        self,
        fn: Callable[Concatenate[InType, P], OutType],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None: ...
    def forward(self, x: InType, **kwargs: P.kwargs) -> OutType: ...
    def extra_repr(self) -> str: ...

ModuleList = EModuleList
ModuleDict = EModuleDict
Sequential = ESequential
ModulePartial = EModulePartial
