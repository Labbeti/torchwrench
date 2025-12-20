#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

from torchwrench.utils.data.sampler import BalancedSampler


class TestCollate(TestCase):
    def test_balanced_sampler(self) -> None:
        indices_per_class = [[0, 1], [2], [3, 4, 5]]

        for num_sampled_per_cls in (0, 10):
            num_classes = len(indices_per_class)

            n_max_iterations = num_classes * num_sampled_per_cls
            sampler = BalancedSampler(indices_per_class, n_max_iterations)

            def find_in_indices(x: int) -> int:
                for i, indices in enumerate(indices_per_class):
                    if x in indices:
                        return i
                return -1

            result = list(sampler)
            result_classes = [find_in_indices(idx) for idx in result]
            expected_classes = list(range(num_classes)) * (
                n_max_iterations // num_classes
            )
            assert result_classes == expected_classes


if __name__ == "__main__":
    unittest.main()
