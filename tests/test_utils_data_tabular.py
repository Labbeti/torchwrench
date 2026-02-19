#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import pythonwrench as pw

import torchwrench as tw
from torchwrench.extras.numpy import NUMPY_AVAILABLE, np
from torchwrench.extras.pandas import PANDAS_AVAILABLE, pd
from torchwrench.extras.speechbrain import SPEECHBRAIN_AVAILABLE, DynamicItemDataset
from torchwrench.utils.data.dataset import TabularDataset


class TestTabularDataset(TestCase):
    def test_dict_list(self) -> None:
        data = {"a": list(range(5)), "b": list(range(5, 10))}
        ds = TabularDataset(data)

        assert len(ds) == 5
        assert ds.shape == (5, 2)
        assert ds.column_names == ds.keys() == ("a", "b")
        assert list(ds.values()) == [list(range(5)), list(range(5, 10))]
        assert ds.to_dict_list() == data
        assert ds.to_list_dict() == pw.dict_list_to_list_dict(data)

        # single_row=T, single_col=T, has_col=T
        assert ds[0, "a"] == 0

        # single_row=T, single_col=T, has_col=F
        # does not exist

        # single_row=T, single_col=F, has_col=T
        assert ds[2, ["a", "b"]] == [2, 7]

        # single_row=T, single_col=F, has_col=F
        assert ds[1] == {"a": 1, "b": 6}

        # single_row=F, single_col=T, has_col=T
        assert ds[[3, 4, 3], "b"] == [8, 9, 8]
        assert ds[:, "a"] == list(range(5))

        # single_row=F, single_col=T, has_col=F
        # does not exist

        # single_row=F, single_col=F, has_col=T
        assert ds[[3, 4, 3], ["b"]] == [[8, 9, 8]]
        assert ds[:, ["a"]] == [list(range(5))]

        # single_row=F, single_col=F, has_col=F
        assert ds[2:4] == [{"a": 2, "b": 7}, {"a": 3, "b": 8}]
        assert ds[[3, 4, 3]] == [{"a": 3, "b": 8}, {"a": 4, "b": 9}, {"a": 3, "b": 8}]

        if NUMPY_AVAILABLE:
            expected = np.array([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])
            assert tw.deep_equal(ds.to_numpy(), expected)

        if PANDAS_AVAILABLE:
            assert tw.deep_equal(ds.to_dataframe(), pd.DataFrame(data))

    def test_ndarray(self) -> None:
        if not NUMPY_AVAILABLE:
            return None

        data: np.ndarray = np.random.rand(10, 3, 2)
        ds = TabularDataset(data)

        result_list_dict = ds.to_list_dict()
        expected_list_dict = [
            {j: data_ij for j, data_ij in enumerate(data_i)} for data_i in data
        ]
        assert tw.deep_equal(result_list_dict, expected_list_dict), (
            f"{result_list_dict=}; {expected_list_dict=}"
        )

        assert tw.deep_equal(ds.to_numpy(), data)
        assert tw.deep_equal(ds[0], data[0])
        assert tw.deep_equal(ds[:2], data[:2])
        assert tw.deep_equal(ds[1, 0], data[1, 0])
        assert tw.deep_equal(ds[5, 1, 0], data[5, 1, 0])
        assert tuple(ds.keys()) == tuple(ds.column_names) == (0, 1, 2)

    def test_dynamic_item_dataset(self) -> None:
        if not SPEECHBRAIN_AVAILABLE:
            return None

        data = {f"{i}": {"string": i} for i in range(100)}
        ds = DynamicItemDataset(data)
        ds.set_output_keys(["string"])
        tabular = TabularDataset(ds)

        assert tabular[0] == {"string": 0}
        assert tabular[:3] == [{"string": i} for i in range(3)]
        assert tabular[3, "string"] == 3
        assert tabular[:3, "string"] == list(range(3))
        assert tabular[:3, ["string"]] == [[0], [1], [2]]

    def test_dynamic_column(self) -> None:
        data = {"a": list(range(5)), "b": list(range(5, 10))}
        ds = TabularDataset(data)

        def double(x):
            return x * 2

        ds.add_dynamic_column(double, requires=("a",), provides="c")

        assert ds[3] == {"a": 3, "b": 8, "c": 6}

        result = ds[4, "c"]
        expected = double(data["a"][4])
        assert result == expected, f"{result=}; {expected=}"

        mask = (
            tw.full((len(ds),), True)
            if not NUMPY_AVAILABLE
            else np.full((len(ds),), True)
        )
        assert ds[mask] == ds.to_list_dict()

    def test_tensor(self) -> None:
        data = tw.rand(10, 2, 3)
        ds = TabularDataset(data)
        mask = tw.rand(data.shape[0]) > 0.5

        assert tw.deep_equal(ds.to_tensor(), data)
        assert tw.deep_equal(ds[mask], data[mask])

    def test_mapping(self) -> None:
        def triple(x):
            return x * 3

        data = {
            "a": ["a1", "a2", "a3"],
            "b": [1, 2, 3],
            "c": tw.arange(1, 4),
        }
        ds = TabularDataset(
            data,
            row_mapper={0: 1, 1: 1},
            col_mapper={"e": "a", "c": "c", "d": "d"},
            fns_list=[(("b",), "d", triple)],
        )

        assert len(ds) == 2
        assert ds.column_names == ("e", "c", "d")

        sample = ds[0]
        expected = {"e": "a2", "c": tw.as_tensor(2), "d": 6}
        assert tw.deep_equal(sample, expected), f"{sample=}; {expected=}"

        datadict = ds.to_dict_list()
        expected = {
            "e": ["a2", "a2"],
            "c": [tw.as_tensor(2), tw.as_tensor(2)],
            "d": [6, 6],
        }
        assert tw.deep_equal(datadict, expected)

        mask = [False] * len(ds)
        sample = ds[mask]
        assert sample == [], f"{sample=}"


if __name__ == "__main__":
    unittest.main()
