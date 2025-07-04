#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchwrench.nn.functional.cropping import crop_dims


class TestCropDims(TestCase):
    def test_crop_dims_example_1(self) -> None:
        x = torch.zeros(10, 10, 10)
        result = crop_dims(x, [1, 2, 3], dims="auto")
        expected = torch.zeros(1, 2, 3)
        assert torch.equal(result, expected)


if __name__ == "__main__":
    unittest.main()
