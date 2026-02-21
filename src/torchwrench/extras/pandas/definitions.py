#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchwrench.core.packaging import _PANDAS_AVAILABLE

if _PANDAS_AVAILABLE:
    import pandas as pd  # noqa: F401  # type: ignore

else:
    pass  # type: ignore
