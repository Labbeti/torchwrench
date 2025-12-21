#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from collections import Counter
from unittest import TestCase

from torchwrench.utils.data.sampler import BalancedSampler


class TestSampler(TestCase):
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
            result_cls_indices = [find_in_indices(idx) for idx in result]

            result_counter = Counter(result_cls_indices)
            expected_counter = Counter(
                {i: n_max_iterations // num_classes for i in range(num_classes)}
                if num_sampled_per_cls > 0
                else {}
            )

            for i in range(0, len(result_cls_indices), num_classes):
                assert set(result_cls_indices[i : i + num_classes]) == set(
                    range(num_classes)
                )
            assert result_counter == expected_counter


if __name__ == "__main__":
    unittest.main()
