#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Type, Union

from typing_extensions import TypeAlias

from torchwrench.core.packaging import _YAML_AVAILABLE

if _YAML_AVAILABLE:
    import yaml  # noqa: F401
    from yaml import (
        BaseLoader,
        FullLoader,
        Loader,
        SafeLoader,
        UnsafeLoader,
    )

    try:
        from yaml import (
            CBaseLoader,
            CFullLoader,
            CLoader,
            CSafeLoader,
            CUnsafeLoader,
        )
    except ImportError:
        from torchwrench.extras.yaml._yaml_fallback import (
            CBaseLoader,
            CFullLoader,
            CLoader,
            CSafeLoader,
            CUnsafeLoader,
        )

else:
    from . import _yaml_fallback as yaml  # noqa: F401
    from ._yaml_fallback import (
        BaseLoader,
        CBaseLoader,
        CFullLoader,
        CLoader,
        CSafeLoader,
        CUnsafeLoader,
        FullLoader,
        Loader,
        SafeLoader,
        UnsafeLoader,
    )


YamlLoaders: TypeAlias = Union[
    Type[Loader],
    Type[BaseLoader],
    Type[FullLoader],
    Type[SafeLoader],
    Type[UnsafeLoader],
    Type[CLoader],
    Type[CBaseLoader],
    Type[CFullLoader],
    Type[CSafeLoader],
    Type[CUnsafeLoader],
]
