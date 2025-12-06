#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch
from torch import Tensor

from torchwrench.nn.modules import (
    Abs,
    Angle,
    AsTensor,
    CropDim,
    CropDims,
    EModuleDict,
    EModuleList,
    EModulePartial,
    ESequential,
    Exp,
    Identity,
    Log,
    LogSoftmaxMultidim,
    Mean,
    PadAndStackRec,
    PadDim,
    PadDims,
    Permute,
    RepeatInterleave,
    RepeatInterleaveNd,
    ResampleNearestRates,
    SoftmaxMultidim,
    ToList,
    Transpose,
    Unsqueeze,
)


class TestSequential(TestCase):
    def test_example_1(self) -> None:
        target_length = 10
        transform = ESequential(
            ResampleNearestRates(0.5),
            PadDim(target_length),
            ResampleNearestRates(2.5),
            CropDim(target_length),
        )

        x = torch.rand(10, 20, target_length)
        result = transform(x)

        assert x.shape == result.shape

    def test_example_2(self) -> None:
        transform = ESequential(
            ToList(),
            AsTensor(),
            Abs(),
            Angle(),
            PadAndStackRec(0.0),
            PadDims([10]),
            CropDims([10]),
            Mean(dim=1),
            Unsqueeze(dim=1),
            Permute(1, 0),
            RepeatInterleaveNd(10, 0),
            Transpose(0, 1),
            LogSoftmaxMultidim(dims=(0, 1)),
            SoftmaxMultidim(dims=(1,)),
            RepeatInterleave(1, -1),
        )

        x = torch.rand(16, 10)
        result = transform(x)

        assert x.shape == result.shape

    def test_example_3(self) -> None:
        transform = ESequential(
            ToList(),
            Identity(),
            AsTensor(),
        )

        x = torch.rand(16, 10)
        result = transform(x)

        assert x.shape == result.shape


class TestModuleDict(TestCase):
    def test_example_1(self) -> None:
        x = torch.rand(10)
        modules = {"log": Log(), "exp": Exp()}
        container = EModuleDict(modules)
        result = container(x)
        assert isinstance(result, dict)


class TestModuleList(TestCase):
    def test_example_1(self) -> None:
        x = torch.rand(10)
        modules = {"log": Log(), "exp": Exp()}
        container = EModuleList(modules.values())
        result = container(x)
        assert isinstance(result, list)


class TestModulePartial(TestCase):
    def test_example_1(self) -> None:
        x = torch.rand(10)
        container = EModulePartial(Log())
        result = container(x)
        assert isinstance(result, Tensor)


if __name__ == "__main__":
    unittest.main()
