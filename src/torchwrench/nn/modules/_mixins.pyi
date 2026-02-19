from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    MutableMapping,
    OrderedDict,
    Protocol,
    TypeVar,
    overload,
)

import torch
from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch import nn
from typing_extensions import TypeAlias

from torchwrench.nn.functional.checksum import checksum_module as checksum_module
from torchwrench.nn.functional.others import count_parameters as count_parameters

T = TypeVar("T", covariant=True)
InType = TypeVar("InType", covariant=False, contravariant=True)
OutType = TypeVar("OutType", covariant=True, contravariant=False)
OutType2 = TypeVar("OutType2", covariant=True, contravariant=False)
OutType3 = TypeVar("OutType3", covariant=False, contravariant=False)
T_MutableMappingStr = TypeVar("T_MutableMappingStr", bound=MutableMapping[str, Any])
DeviceDetectMode: Incomplete
DEVICE_DETECT_MODES: Incomplete
pylog: Incomplete

class SupportsTypedForward(Protocol[InType, OutType]):
    def __call__(self, *args, **kwargs): ...
    def forward(self, x: InType, /) -> OutType: ...

TypedModuleLike: TypeAlias

class ProxyDeviceModule(nn.Module):
    def __init__(self, *, device_detect_mode: DeviceDetectMode = ...) -> None: ...
    @property
    def device_detect_mode(self) -> DeviceDetectMode: ...
    def get_device(self) -> torch.device | None: ...
    def get_devices(
        self,
        *,
        params: bool = True,
        buffers: bool = True,
        recurse: bool = True,
        output_type: Callable[[Iterator[torch.device]], T] = ...,
    ) -> T: ...

class ConfigModule(nn.Module, Generic[T_MutableMappingStr]):
    def __init__(
        self,
        *,
        strict_load: bool = False,
        config_to_extra_repr: bool | None = None,
        config: T_MutableMappingStr | None = None,
    ) -> None: ...
    @property
    def config(self) -> T_MutableMappingStr: ...
    def __setattr__(self, name: str, value: Any) -> None: ...
    def __delattr__(self, name) -> None: ...
    def extra_repr(self) -> str: ...
    def add_module(self, name: str, module: nn.Module | None) -> None: ...
    def get_extra_state(self) -> Any: ...
    def set_extra_state(self, state: Any) -> Any: ...

class TypedModule(nn.Module, Generic[InType, OutType]):
    def __call__(self, *args: InType, **kwargs: InType) -> OutType: ...

class TypedSequential(
    TypedModule[InType, OutType], nn.Sequential, Generic[InType, OutType]
):
    def __init__(
        self, *args, unpack_tuple: bool = False, unpack_dict: bool = False
    ) -> None: ...
    @property
    def unpack_tuple(self) -> bool: ...
    @property
    def unpack_dict(self) -> bool: ...
    def __call__(self, x: InType) -> OutType: ...
    def forward(self, x: InType) -> OutType: ...
    def tolist(self) -> list[nn.Module]: ...
    def todict(self) -> dict[str, nn.Module]: ...

class EModule(
    ConfigModule,
    TypedModule[InType, OutType],
    ProxyDeviceModule,
    Generic[InType, OutType],
):
    def __init__(
        self,
        *,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = ...,
    ) -> None: ...
    def count_parameters(
        self,
        *,
        recurse: bool = True,
        only_trainable: bool = False,
        buffers: bool = False,
    ) -> int: ...
    def checksum(
        self,
        *,
        only_trainable: bool = False,
        with_names: bool = False,
        buffers: bool = False,
        training: bool = False,
    ) -> int: ...
    @overload
    def chain(
        self, *others: TypedModuleLike[Any, OutType]
    ) -> ESequential[InType, OutType]: ...
    @overload
    def chain(self, *others: nn.Module) -> ESequential[InType, Any]: ...
    def __or__(
        self, other: TypedModuleLike[Any, OutType]
    ) -> ESequential[InType, OutType]: ...
    def __ror__(
        self, other: TypedModuleLike[InType, Any]
    ) -> ESequential[InType, OutType]: ...

class ESequential(
    EModule[InType, OutType], TypedSequential[InType, OutType], Generic[InType, OutType]
):
    @overload
    def __init__(
        self,
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        arg0: TypedModuleLike[InType, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = ...,
        unpack_dict: bool = False,
    ) -> None: ...
    @overload
    def __init__(
        self,
        arg0: TypedModuleLike[InType, Any],
        arg1: TypedModuleLike[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        arg0: TypedModuleLike[InType, Any],
        arg1: TypedModuleLike[Any, Any],
        arg2: TypedModuleLike[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        arg0: TypedModuleLike[InType, Any],
        arg1: TypedModuleLike[Any, Any],
        arg2: TypedModuleLike[Any, Any],
        arg3: TypedModuleLike[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        arg0: TypedModuleLike[InType, Any],
        arg1: TypedModuleLike[Any, Any],
        arg2: TypedModuleLike[Any, Any],
        arg3: TypedModuleLike[Any, Any],
        arg4: TypedModuleLike[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        arg0: TypedModuleLike[InType, Any],
        arg1: TypedModuleLike[Any, Any],
        arg2: TypedModuleLike[Any, Any],
        arg3: TypedModuleLike[Any, Any],
        arg4: TypedModuleLike[Any, Any],
        arg5: TypedModuleLike[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        arg0: TypedModuleLike[InType, Any],
        arg1: TypedModuleLike[Any, Any],
        arg2: TypedModuleLike[Any, Any],
        arg3: TypedModuleLike[Any, Any],
        arg4: TypedModuleLike[Any, Any],
        arg5: TypedModuleLike[Any, Any],
        arg6: TypedModuleLike[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        arg0: TypedModuleLike[InType, Any],
        arg1: TypedModuleLike[Any, Any],
        arg2: TypedModuleLike[Any, Any],
        arg3: TypedModuleLike[Any, Any],
        arg4: TypedModuleLike[Any, Any],
        arg5: TypedModuleLike[Any, Any],
        arg6: TypedModuleLike[Any, Any],
        arg7: TypedModuleLike[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        arg0: TypedModuleLike[InType, Any],
        arg1: TypedModuleLike[Any, Any],
        arg2: TypedModuleLike[Any, Any],
        arg3: TypedModuleLike[Any, Any],
        arg4: TypedModuleLike[Any, Any],
        arg5: TypedModuleLike[Any, Any],
        arg6: TypedModuleLike[Any, Any],
        arg7: TypedModuleLike[Any, Any],
        arg8: TypedModuleLike[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        arg0: TypedModuleLike[InType, Any],
        arg1: TypedModuleLike[Any, Any],
        arg2: TypedModuleLike[Any, Any],
        arg3: TypedModuleLike[Any, Any],
        arg4: TypedModuleLike[Any, Any],
        arg5: TypedModuleLike[Any, Any],
        arg6: TypedModuleLike[Any, Any],
        arg7: TypedModuleLike[Any, Any],
        arg8: TypedModuleLike[Any, Any],
        arg9: TypedModuleLike[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        arg: OrderedDict[str, TypedModuleLike[InType, OutType]],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        arg: OrderedDict[str, nn.Module],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        *args: nn.Module,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = ...,
    ) -> None: ...
