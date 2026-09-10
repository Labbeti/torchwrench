#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .definitions import (  # noqa: F401  # type: ignore
        pandas,
        pandas_is_available,
        pd,
    )
    from .pandas import (
        concat_without_duplicated_col_names,
        drop_duplicated_col_names,
        empty_dataframe,
    )

else:
    import lazy_loader as lazy

    __getattr__, __dir__, __all__ = lazy.attach(
        __name__,
        submodules=["definitions", "pandas"],
        submod_attrs={
            "definitions": ["pandas_is_available", "pandas", "pd"],
            "pandas": [
                "empty_dataframe",
                "drop_duplicated_col_names",
                "concat_without_duplicated_col_names",
            ],
        },
    )


del TYPE_CHECKING
