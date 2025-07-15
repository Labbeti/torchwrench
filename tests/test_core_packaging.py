#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

from torchwrench.core.packaging import requires_packages


class TestPackaging(TestCase):
    def test_requires_packages(self) -> None:
        @requires_packages("torch")
        def f(x: int) -> int:
            return x

        @requires_packages("not_exist")
        def g(x: int) -> int:
            return x

        _ = f(1)
        with self.assertRaises(ImportError):
            _ = g(1)


if __name__ == "__main__":
    unittest.main()
