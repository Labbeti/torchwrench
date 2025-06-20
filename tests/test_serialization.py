#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import unittest
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest import TestCase

import pythonwrench as pw
import torch

import torchwrench as tw
from torchwrench.extras import _NUMPY_AVAILABLE, _SAFETENSORS_AVAILABLE, _YAML_AVAILABLE
from torchwrench.hub.paths import get_tmp_dir
from torchwrench.nn.functional import deep_equal
from torchwrench.serialization.common import (
    SavingBackend,
    _fpath_to_saving_backend,
    to_builtin,
)


class TestSaving(TestCase):
    def test_examples(self) -> None:
        x = [
            [
                torch.arange(3)[None],
                "a",
                Path("./path"),
                Counter(["a", "b", "a", "c", "a"]),
                (),
            ],
        ]
        expected = [[[list(range(3))], "a", "path", {"a": 3, "b": 1, "c": 1}, []]]
        result = to_builtin(x)
        assert result == expected, f"{result=}; {expected=}"

    def test_backend(self) -> None:
        tests: List[Tuple[str, SavingBackend]] = [
            ("test.json", "json"),
            ("test.json.yaml", "yaml"),
            ("test.yaml.json", "json"),
        ]

        for fpath, expected_backend in tests:
            backend = _fpath_to_saving_backend(fpath)
            assert backend == expected_backend

    def test_save_load(self) -> None:
        n = 1
        data1 = {
            "arange": tw.arange(n),
            "full": tw.full((n, 5), 9),
            "ones": tw.ones(n, 5),
            "rand": tw.rand(n),
            "randint": tw.randint(0, 100, (n,)),
            "randperm": tw.randperm(n),
            "zeros": tw.zeros(n, 1),
            "empty": tw.empty(n, 2),
        }
        data2: Dict[str, Any] = copy.copy(data1)
        data2.update({"randstr": [pw.randstr(2) for _ in range(n)]})

        assert pw.is_full(map(len, data2.values()))

        tests: List[Tuple[SavingBackend, Any, bool, dict]] = [
            ("json", data2, True, dict()),
            ("pickle", data2, False, dict()),
        ]

        if _NUMPY_AVAILABLE:
            from torchwrench.extras.numpy import to_ndarray

            added_tests: List[Tuple[SavingBackend, Any, bool, dict]] = [
                ("numpy", to_ndarray(v), False, dict()) for k, v in data2.items()
            ]
            tests += added_tests

        if _SAFETENSORS_AVAILABLE:
            added_tests: List[Tuple[SavingBackend, Any, bool, dict]] = [
                ("safetensors", data1, False, dict()),
            ]
            tests += added_tests

        if _YAML_AVAILABLE:
            added_tests: List[Tuple[SavingBackend, Any, bool, dict]] = [
                ("yaml", data1, True, dict()),
                ("yaml", data2, True, dict()),
            ]
            tests += added_tests

        for i, (backend, data, to_builtins, load_kwds) in enumerate(tests):
            if to_builtins:
                data = tw.to_builtin(data)
            if backend == "safetensors":
                data = pw.sorted_dict(data)

            fpath = get_tmp_dir().joinpath(f"tmp.{backend}")
            tw.dump(data, fpath, saving_backend=backend)
            result = tw.load(fpath, saving_backend=backend, **load_kwds)

            assert deep_equal(data, result), f"{backend=}, {i=}/{len(tests)}"


if __name__ == "__main__":
    unittest.main()
