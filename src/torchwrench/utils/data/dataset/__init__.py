#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .slicer import DatasetSlicer, DatasetSlicerWrapper
    from .tabular import TabularDataset
    from .wrapper import (
        EmptyDataset,
        IterableSubset,
        IterableTransformWrapper,
        IterableWrapper,
        Subset,
        TransformWrapper,
        Wrapper,
    )

else:
    import lazy_loader as lazy

    __getattr__, __dir__, __all__ = lazy.attach(
        __name__,
        submodules=["slicer", "tabular", "wrapper"],
        submod_attrs={
            "slicer": ["DatasetSlicer", "DatasetSlicerWrapper"],
            "tabular": ["TabularDataset"],
            "wrapper": [
                "EmptyDataset",
                "IterableSubset",
                "IterableTransformWrapper",
                "IterableWrapper",
                "Subset",
                "TransformWrapper",
                "Wrapper",
            ],
        },
    )

del TYPE_CHECKING
