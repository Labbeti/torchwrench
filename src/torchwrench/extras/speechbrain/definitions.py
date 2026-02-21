#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchwrench.core.packaging import _SPEECHBRAIN_AVAILABLE

if _SPEECHBRAIN_AVAILABLE:
    from speechbrain.dataio.dataset import (
        DynamicItemDataset,  # noqa: F401  # type: ignore
    )
else:
    pass  # type: ignore
