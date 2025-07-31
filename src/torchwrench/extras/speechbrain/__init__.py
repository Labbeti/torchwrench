#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchwrench.core.packaging import _SPEECHBRAIN_AVAILABLE

if _SPEECHBRAIN_AVAILABLE:
    from speechbrain.dataio.dataset import DynamicItemDataset

else:
    from torchwrench.extras.speechbrain._speechbrain_fallback import DynamicItemDataset


__all__ = [
    "_SPEECHBRAIN_AVAILABLE",
    "DynamicItemDataset",
]
