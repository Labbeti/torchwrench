#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Type, Union

from typing_extensions import TypeAlias

from torchwrench.core.packaging import _YAML_AVAILABLE

if not _YAML_AVAILABLE:
    from . import _yaml_fallback as yaml  # noqa: F401
    from ._yaml_fallback import (  # noqa: F401
        BaseLoader,
        CBaseLoader,
        CFullLoader,
        CLoader,
        CSafeLoader,
        CUnsafeLoader,
        FullLoader,
        Loader,
        MappingNode,
        Node,
        ParserError,
        SafeLoader,
        ScalarNode,
        ScannerError,
        SequenceNode,
        UnsafeLoader,
    )

else:
    import yaml  # noqa: F401
    from yaml import (  # noqa: F401
        BaseLoader,
        FullLoader,
        Loader,
        MappingNode,
        Node,
        SafeLoader,
        ScalarNode,
        SequenceNode,
        UnsafeLoader,
    )
    from yaml.parser import ParserError  # noqa: F401
    from yaml.scanner import ScannerError  # noqa: F401

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
