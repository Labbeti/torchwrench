#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, Union

from pythonwrench.collections import dump_dict
from torch import Tensor
from torch.types import Number

from torchwrench.core.make import DeviceLike, DTypeLike, GeneratorLike
from torchwrench.nn.functional.padding import (
    PadAlign,
    PadMode,
    PadValue,
    pad_and_stack_rec,
    pad_dim,
    pad_dims,
)

from .module import Module


class PadDim(Module):
    """
    For more information, see :func:`~torchwrench.nn.functional.pad.pad_dim`.
    """

    def __init__(
        self,
        target_length: int,
        *,
        dim: int = -1,
        align: PadAlign = "left",
        pad_value: PadValue = 0.0,
        mode: PadMode = "constant",
        generator: GeneratorLike = None,
    ) -> None:
        super().__init__()
        self.target_length = target_length
        self.dim = dim
        self.align: PadAlign = align
        self.pad_value = pad_value
        self.mode: PadMode = mode
        self.generator: GeneratorLike = generator

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return pad_dim(
            x,
            target_length=self.target_length,
            dim=self.dim,
            align=self.align,
            pad_value=self.pad_value,
            mode=self.mode,
            generator=self.generator,
        )

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                target_length=self.target_length,
                dim=self.dim,
                align=self.align,
                pad_value=self.pad_value,
                mode=self.mode,
            )
        )


class PadDims(Module):
    """
    For more information, see :func:`~torchwrench.nn.functional.pad.pad_dims`.
    """

    def __init__(
        self,
        target_lengths: Iterable[int],
        *,
        dims: Iterable[int] = (-1,),
        aligns: Iterable[PadAlign] = ("left",),
        pad_value: PadValue = 0.0,
        mode: PadMode = "constant",
        generator: GeneratorLike = None,
    ) -> None:
        super().__init__()
        self.target_lengths = target_lengths
        self.aligns = aligns
        self.pad_value = pad_value
        self.dims = dims
        self.mode: PadMode = mode
        self.generator: GeneratorLike = generator

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return pad_dims(
            x,
            target_lengths=self.target_lengths,
            dims=self.dims,
            aligns=self.aligns,
            pad_value=self.pad_value,
            mode=self.mode,
            generator=self.generator,
        )

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                target_lengths=self.target_lengths,
                dims=self.dims,
                aligns=self.aligns,
                pad_value=self.pad_value,
                mode=self.mode,
            )
        )


class PadAndStackRec(Module):
    """
    For more information, see :func:`~torchwrench.nn.functional.pad.pad_and_stack_rec`.
    """

    def __init__(
        self,
        pad_value: Number = 0,
        *,
        align: PadAlign = "left",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.pad_value = pad_value
        self.align: PadAlign = align
        self.device = device
        self.dtype = dtype

    def forward(
        self,
        sequence: Union[Tensor, int, float, tuple, list],
    ) -> Tensor:
        return pad_and_stack_rec(
            sequence,
            self.pad_value,
            align=self.align,
            device=self.device,
            dtype=self.dtype,
        )

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                pad_value=self.pad_value,
                align=self.align,
                device=self.device,
                dtype=self.dtype,
            )
        )
