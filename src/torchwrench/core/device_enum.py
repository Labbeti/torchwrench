#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import auto

from pythonwrench.enum import StrEnum


class DeviceEnum(StrEnum):
    cuda = auto()
    cpu = auto()


# TODO: rm
# from enum import Enum


# class _DeviceType:
#     ...


# class CUDADeviceType(_DeviceType):
#     ...

# class CPUDeviceType(_DeviceType):
#     ...


# class DeviceEnum(Enum):
#     cuda = CUDADeviceType
#     cpu = CPUDeviceType
