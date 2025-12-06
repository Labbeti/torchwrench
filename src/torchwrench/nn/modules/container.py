#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from typing import Callable, Dict, Generic, Iterable, List, Mapping, Optional, overload

from torch import nn
from typing_extensions import Concatenate, ParamSpec

from torchwrench.core.config import _REPLACE_MODULE_CLASSES

from ._mixins import (
    _DEFAULT_DEVICE_DETECT_MODE,
    ConfigModule,
    DeviceDetectMode,
    EModule,  # noqa: F401
    ESequential,  # noqa: F401
    InType,
    OutType,
    OutType3,
    TypedModuleLike,
)

P = ParamSpec("P")


class EModuleList(
    Generic[InType, OutType3],
    EModule[InType, List[OutType3]],
    nn.ModuleList,
):
    """Enriched torch.nn.ModuleList with proxy device, forward typing and automatic configuration detection from attributes.

    Designed to work with `torchwrench.nn.EModule` instances.
    The default behaviour is the same than PyTorch ModuleList class, except for the forward call which returns a list containing the output of each module called separately.
    """

    @overload
    def __init__(
        self,
        modules: Optional[Iterable[TypedModuleLike[InType, OutType3]]] = None,
        *,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
    ) -> None: ...

    @overload
    def __init__(
        self,
        modules: Optional[Iterable[nn.Module]] = None,
        *,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
    ) -> None: ...

    def __init__(
        self,
        modules=None,
        *,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
    ) -> None:
        EModule.__init__(
            self,
            strict_load=strict_load,
            config_to_extra_repr=config_to_extra_repr,
            device_detect_mode=device_detect_mode,
        )
        nn.ModuleList.__init__(self, modules)

    def forward(self, *args: InType, **kwargs: InType) -> List[OutType3]:
        return [module(*args, **kwargs) for module in self]


class EModuleDict(
    Generic[InType, OutType3],
    EModule[InType, Dict[str, OutType3]],
    nn.ModuleDict,
):
    """Enriched torch.nn.ModuleDict with proxy device, forward typing and automatic configuration detection from attributes.

    Designed to work with `torchwrench.nn.EModule` instances.
    The default behaviour is the same than PyTorch ModuleDict class, except for the forward call which returns a dict containing the output of each module called separately.
    """

    @overload
    def __init__(
        self,
        modules: Optional[Mapping[str, TypedModuleLike[InType, OutType3]]] = None,
        *,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
    ) -> None: ...

    @overload
    def __init__(
        self,
        modules: Optional[Mapping[str, nn.Module]] = None,
        *,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
    ) -> None: ...

    def __init__(
        self,
        modules=None,
        *,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
    ) -> None:
        EModule.__init__(
            self,
            strict_load=strict_load,
            config_to_extra_repr=config_to_extra_repr,
            device_detect_mode=device_detect_mode,
        )
        nn.ModuleDict.__init__(self, modules)

    def forward(self, *args: InType, **kwargs: InType) -> Dict[str, OutType3]:
        return {name: module(*args, **kwargs) for name, module in self.items()}  # type: ignore


class EModulePartial(
    Generic[InType, OutType],
    EModule[InType, OutType],
):
    """Wrap a python callable to nn.Module class."""

    def __init__(
        self,
        fn: Callable[Concatenate[InType, P], OutType],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """
        Args:
            fn: Callable to wrap.
            *args: Positional arguments.
            **kwargs:
        """
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: InType, **kwargs: P.kwargs) -> OutType:  # type: ignore
        kwds = copy.copy(self.kwargs)
        kwds.update(kwargs)
        return self.fn(x, *self.args, **kwds)  # type: ignore

    def extra_repr(self) -> str:
        return f"{self.fn.__name__}, {ConfigModule.extra_repr(self)}"


if _REPLACE_MODULE_CLASSES:
    ModuleList = EModuleList
    ModuleDict = EModuleDict
    Sequential = ESequential
else:
    from torch.nn import ModuleDict, ModuleList, Sequential  # noqa: F401

ModulePartial = EModulePartial
