#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchwrench.core.packaging import _PANDAS_AVAILABLE

if not _PANDAS_AVAILABLE:
    from . import _pandas_fallback as pandas  # noqa: F401  # type: ignore
    from . import _pandas_fallback as pd  # noqa: F401  # type: ignore

else:
    import pandas  # noqa: F401  # type: ignore
    import pandas as pd  # noqa: F401  # type: ignore
