#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchwrench.core.packaging import PANDAS_AVAILABLE

if not PANDAS_AVAILABLE:
    from torchwrench.extras.pandas import _pandas_fallback as pd

else:
    import pandas as pd


__all__ = [
    "PANDAS_AVAILABLE",
    "pd",
]
