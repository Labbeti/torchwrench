#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    TypedModule,
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
    def __init__(
        self,
        fn: Callable[Concatenate[InType, P], OutType],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: InType) -> OutType:  # type: ignore
        return self.fn(x, *self.args, **self.kwargs)

    def extra_repr(self) -> str:
        return f"{self.fn.__name__}, {ConfigModule.extra_repr(self)}"


if _REPLACE_MODULE_CLASSES:
    ModuleList = EModuleList
    ModuleDict = EModuleDict
    Sequential = ESequential
else:
    from torch.nn import ModuleDict, ModuleList, Sequential  # noqa: F401

ModulePartial = EModulePartial


def __test_typing_1() -> None:
    import torch
    from torch import Tensor

    class LayerA(EModule[Tensor, Tensor]):
        def forward(self, x: Tensor) -> Tensor:
            return x * x

    class LayerB(EModule[Tensor, int]):
        def forward(self, x: Tensor) -> int:
            return int(x.sum().item())

    class LayerC(EModule[int, Tensor]):
        def forward(self, x: int) -> Tensor:
            return torch.as_tensor(x)

    x = torch.rand(10)
    xa = LayerA()(x)
    xb = LayerB()(x)

    seq = ESequential(LayerA(), LayerA(), LayerB())
    xab = seq(x)

    seq = LayerA() | LayerA() | LayerB()
    xab = seq(x)

    seq = LayerC().chain(LayerA())
    xc = seq(2)

    assert isinstance(xa, Tensor)
    assert isinstance(xb, int)
    assert isinstance(xab, int)
    assert isinstance(xc, Tensor)

    class LayerD(nn.Module):
        def forward(self, x: Tensor) -> int:
            return int(x.item())

    class LayerE(nn.Module):
        def forward(self, x: bool) -> str:
            return str(x)

    seq = ESequential(LayerD(), LayerE())
    y = seq(torch.rand())

    assert isinstance(y, str)

    class LayerF(TypedModule[bool, str]):
        def forward(self, x):
            return str(x)

    seq = ESequential(LayerF())
    y = seq(True)
