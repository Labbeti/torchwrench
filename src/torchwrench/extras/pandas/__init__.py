#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchwrench.core.packaging import _PANDAS_AVAILABLE

if _PANDAS_AVAILABLE:
    import pandas as pd

else:
    from torchwrench.extras.pandas import _pandas_fallback as pd


__all__ = [
    "_PANDAS_AVAILABLE",
    "pd",
]
