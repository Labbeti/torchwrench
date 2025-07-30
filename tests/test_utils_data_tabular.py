#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torchwrench as tw
from torchwrench.core.packaging import _PANDAS_AVAILABLE
from torchwrench.extras.pandas import pd
from torchwrench.utils.data.tabular import TabularDataset


class TestSplit(TestCase):
    def test_dict_list(self) -> None:
        data = {"a": list(range(5)), "b": list(range(5, 10))}
        ds = TabularDataset(data)

        assert len(ds) == 5
        assert ds.column_names == ds.keys() == ("a", "b")
        assert ds.values() == [list(range(5)), list(range(5, 10))]
        assert ds.to_dict() == data

        if not _PANDAS_AVAILABLE:
            return None
        assert tw.deep_equal(ds.to_dataframe(), pd.DataFrame(data))
        assert ds.shape == (5, 2)


if __name__ == "__main__":
    unittest.main()
