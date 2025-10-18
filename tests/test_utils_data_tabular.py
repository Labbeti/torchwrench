#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import pythonwrench as pw
from speechbrain.dataio.dataset import DynamicItemDataset

import torchwrench as tw
from torchwrench.extras.numpy import _NUMPY_AVAILABLE, np
from torchwrench.extras.pandas import _PANDAS_AVAILABLE, pd
from torchwrench.utils.data.tabular.dataset2 import TabularDataset


class TestTabularDataset(TestCase):
    def test_dict_list(self) -> None:
        data = {"a": list(range(5)), "b": list(range(5, 10))}
        ds = TabularDataset(data)

        assert len(ds) == 5
        assert ds.shape == (5, 2)
        assert ds.column_names == ds.keys() == ("a", "b")
        assert ds.values() == [list(range(5)), list(range(5, 10))]
        assert ds.to_dict_list() == data
        assert ds.to_list_dict() == pw.dict_list_to_list_dict(data)

        assert ds[1] == {"a": 1, "b": 6}
        assert ds[2:4] == [{"a": 2, "b": 7}, {"a": 3, "b": 8}]
        assert ds[[3, 4, 3]] == [{"a": 3, "b": 8}, {"a": 4, "b": 9}, {"a": 3, "b": 8}]
        assert ds[[3, 4, 3], "b"] == [8, 9, 8]
        assert ds[[3, 4, 3], ["b"]] == [[8, 9, 8]]
        assert ds[:, "a"] == list(range(5))
        assert ds[:, ["a"]] == [list(range(5))]

        if _NUMPY_AVAILABLE:
            expected = np.array([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])
            assert tw.deep_equal(ds.to_numpy(), expected)

        if _PANDAS_AVAILABLE:
            assert tw.deep_equal(ds.to_dataframe(), pd.DataFrame(data))

    def test_ndarray(self) -> None:
        if not _NUMPY_AVAILABLE:
            return None

        data: np.ndarray = np.random.rand(10, 3, 2)
        ds = TabularDataset(data)

        assert tw.deep_equal(ds.to_numpy(), data)
        assert tw.deep_equal(ds[0], data[0])
        assert tw.deep_equal(ds[:2], data[:2])
        assert tw.deep_equal(ds[1, 0], data[1, 0])
        assert tw.deep_equal(ds[5, 1, 0], data[5, 1, 0])
        assert tuple(ds.keys()) == tuple(ds.column_names) == (0, 1, 2)

    def test_dynamic_item_dataset(self) -> None:
        data = {f"{i}": {"string": i} for i in range(100)}
        ds = DynamicItemDataset(data)
        ds.set_output_keys(["string"])
        tabular = TabularDataset(ds)

        assert tabular[0] == {"string": 0}
        assert tabular[:3] == [{"string": i} for i in range(3)]
        assert tabular[3, "string"] == 3
        assert tabular[:3, "string"] == list(range(3))
        assert tabular[:3, ["string"]] == [list(range(3))]

    def test_dynamic_column(self) -> None:
        data = {"a": list(range(5)), "b": list(range(5, 10))}
        ds = TabularDataset(data)

        def double(x):
            return x * 2

        ds.add_dynamic_column(double, requires=("a",), provides=("c",))
        # ds.add_output_keys(("c",))

        assert ds[3] == {"a": 3, "b": 8, "c": 6}
        assert tw.deep_equal(ds[4, "c"], data["a"][4] * 2)


if __name__ == "__main__":
    unittest.main()
