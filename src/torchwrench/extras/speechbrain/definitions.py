#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchwrench.core.packaging import _SPEECHBRAIN_AVAILABLE

if not _SPEECHBRAIN_AVAILABLE:
    from ._speechbrain_fallback import DynamicItemDataset  # noqa: F401  # type: ignore

else:
    from speechbrain.dataio.dataset import (  # noqa: F401  # type: ignore
        DynamicItemDataset,
    )
