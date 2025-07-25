#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchwrench.nn.functional.multiclass import (
    index_to_name,
    index_to_onehot,
    onehot_to_index,
)


class TestMulticlass(TestCase):
    def test_index_to_name(self) -> None:
        indices = torch.as_tensor([1, 2, 0, 0])
        mapping = ["a", "b", "c"]
        expected = ["b", "c", "a", "a"]
        assert index_to_name(indices, mapping) == expected

    def test_index_to_onehot_1(self) -> None:
        indices = torch.as_tensor([[0, 2, 1], [0, 0, 2]])
        expected = torch.as_tensor(
            [[[1, 0, 0], [0, 0, 1], [0, 1, 0]], [[1, 0, 0], [1, 0, 0], [0, 0, 1]]],
            dtype=torch.bool,
        )
        result = index_to_onehot(indices, 3)
        assert torch.equal(result, expected)

    def test_index_to_onehot_2(self) -> None:
        indices = torch.as_tensor([[0, -1, 1], [0, 0, 2], [-1, -1, -1]])
        expected = torch.as_tensor(
            [
                [[1, 0, 0], [0, 0, 0], [0, 1, 0]],
                [[1, 0, 0], [1, 0, 0], [0, 0, 1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            dtype=torch.bool,
        )
        result = index_to_onehot(indices, 3, padding_idx=-1)
        assert torch.equal(result, expected)

    def test_onehot_to_index_3d(self) -> None:
        onehot = torch.as_tensor([[[0, 1, 0], [1, 0, 0]], [[0, 0, 1], [0, 0, 1]]])
        expected = torch.as_tensor([[1, 0], [2, 2]])
        result = onehot_to_index(onehot)
        assert torch.equal(result, expected)

    def test_index_to_onehot_3d(self) -> None:
        indices = [[1, 0], [2, 2]]
        expected = torch.as_tensor(
            [[[0, 1, 0], [1, 0, 0]], [[0, 0, 1], [0, 0, 1]]]
        ).bool()
        result = index_to_onehot(indices, 3)
        assert torch.equal(result, expected)

    def test_index_to_onehot_padding_idx(self) -> None:
        x = torch.randint(0, 100, (1000,))
        mask = torch.rand_like(x, dtype=torch.float) > 0.2
        x = torch.where(mask, -10, x)
        onehots = index_to_onehot(x, 100, padding_idx=-10)
        assert onehots[mask].eq(False).all()

        index = onehot_to_index(onehots, padding_idx=-10)
        assert torch.equal(x, index)


if __name__ == "__main__":
    unittest.main()
