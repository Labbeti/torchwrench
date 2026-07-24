#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchwrench.core.packaging import pandas_is_available

if not pandas_is_available():
    from . import _pandas_fallback as pandas  # noqa: F401  # type: ignore
    from . import _pandas_fallback as pd  # noqa: F401  # type: ignore

else:
    import pandas  # noqa: F401  # type: ignore
    import pandas as pd  # noqa: F401  # type: ignore
