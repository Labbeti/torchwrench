#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchwrench.core.packaging import speechbrain_is_available

if not speechbrain_is_available():
    from ._speechbrain_fallback import DynamicItemDataset  # noqa: F401  # type: ignore
else:
    from speechbrain.dataio.dataset import (
        DynamicItemDataset,  # noqa: F401  # type: ignore
    )
