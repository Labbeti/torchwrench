#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .collate import AdvancedCollateDict, CollateDict
    from .dataloader import get_auto_num_cpus, get_auto_num_gpus
    from .dataset.slicer import DatasetSlicer, DatasetSlicerWrapper
    from .dataset.wrapper import (
        EmptyDataset,
        IterableSubset,
        IterableTransformWrapper,
        IterableWrapper,
        Subset,
        TransformWrapper,
        Wrapper,
    )
    from .sampler import BalancedSampler, SubsetCycleSampler, SubsetSampler
    from .split import balanced_monolabel_split, random_split

else:
    import lazy_loader as lazy

    __getattr__, __dir__, __all__ = lazy.attach(
        __name__,
        submodules=["collate", "dataloader", "dataset", "sampler", "split"],
        submod_attrs={
            "collate": ["AdvancedCollateDict", "CollateDict"],
            "dataloader": ["get_auto_num_cpus", "get_auto_num_gpus"],
            "dataset": ["slicer", "wrapper"],
            "sampler": ["BalancedSampler", "SubsetCycleSampler", "SubsetSampler"],
            "split": ["balanced_monolabel_split", "random_split"],
        },
    )
