#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

# from enum import auto

# from pythonwrench.enum import StrEnum


# class DeviceEnum(StrEnum):
#     cuda = auto()
#     cpu = auto()


# TODO: rm
# from enum import Enum


class DeviceBase: ...


class CUDADeviceType(DeviceBase): ...


class CPUDeviceType(DeviceBase): ...


# class DeviceEnum(Enum):
#     cuda = CUDADeviceType
#     cpu = CPUDeviceType


def device_cls_to_torch_device(device_cls: DeviceBase) -> torch.device:
    if device_cls is CUDADeviceType:
        return torch.device("cuda")
    elif device_cls is CPUDeviceType:
        return torch.device("cpu")
    else:
        raise ValueError(f"Invalid argument {device_cls=}.")
