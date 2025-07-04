#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import unittest
import warnings
from unittest import TestCase
from urllib.error import HTTPError

from torch import Tensor

from torchwrench.hub.download import _get_filename_from_url
from torchwrench.hub.paths import get_tmp_dir
from torchwrench.hub.registry import RegistryHub


class TestFilename(TestCase):
    def test_filename_from_url(self) -> None:
        tests = [
            (
                "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json",
                "kinetics_classnames.json",
            ),
            (
                "https://zenodo.org/record/8020843/files/convnext_tiny_465mAP_BL_AC_70kit.pth?download=1",
                "convnext_tiny_465mAP_BL_AC_70kit.pth",
            ),
            ("random.test", "random.test"),
        ]

        for input_, expected in tests:
            result = _get_filename_from_url(input_)
            assert result == expected


class TestRegistryHub(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tmpdir = get_tmp_dir().joinpath("torchwrench_tests")
        tmpdir.mkdir(parents=True, exist_ok=True)
        cls.tmpdir = tmpdir

    def test_cnext_register(self) -> None:
        register = RegistryHub(
            infos={
                "cnext_bl_70": {
                    "architecture": "ConvNeXt",
                    "url": "https://zenodo.org/record/8020843/files/convnext_tiny_465mAP_BL_AC_70kit.pth?download=1",
                    "hash_value": "0688ae503f5893be0b6b71cb92f8b428",
                    "hash_type": "md5",
                    "fname": "convnext_tiny_465mAP_BL_AC_70kit.pth",
                    "state_dict_key": "model",
                },
            },
            register_root=self.tmpdir,
        )

        model_name = "cnext_bl_70"
        num_tries = 5
        for i in range(num_tries):
            try:
                register.download_file(model_name, force=False)
                break
            except HTTPError as err:
                if i + 1 < num_tries:
                    msg = f"Failed to download {i + 1} times."
                    warnings.warn(msg)
                    raise err
                else:
                    msg = f"Found error {err} at try number {i + 1}/{num_tries}, retrying..."
                    warnings.warn(msg)
                    time.sleep(i + 1)

        state_dict = register.load_state_dict(
            model_name,
            offline=True,
            load_kwds=dict(map_location="cpu", weights_only=False),
        )

        assert isinstance(state_dict, dict)
        assert all(isinstance(k, str) for k in state_dict.keys())
        assert all(isinstance(v, Tensor) for v in state_dict.values())


if __name__ == "__main__":
    unittest.main()
