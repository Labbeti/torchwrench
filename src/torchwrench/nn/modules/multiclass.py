#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Generic, List, Mapping, Optional, Sequence, Union

import torch
from pythonwrench.collections import dump_dict
from torch import Tensor, nn

from torchwrench.core.make import DeviceLike, DTypeLike
from torchwrench.nn.functional.multiclass import (
    T_Name,
    index_to_name,
    index_to_onehot,
    name_to_index,
    name_to_onehot,
    onehot_to_index,
    onehot_to_name,
    probs_to_index,
    probs_to_name,
    probs_to_onehot,
)

from .module import Module


class IndexToOnehot(Module):
    """
    For more information, see :func:`~torchwrench.nn.functional.multiclass.index_to_onehot`.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        padding_idx: Optional[int] = None,
        device: DeviceLike = None,
        dtype: DTypeLike = torch.bool,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.padding_idx = padding_idx
        self.device = device
        self.dtype = dtype

    def forward(
        self,
        index: Union[List[int], Tensor],
    ) -> Tensor:
        onehot = index_to_onehot(
            index,
            self.num_classes,
            padding_idx=self.padding_idx,
            device=self.device,
            dtype=self.dtype,
        )
        return onehot

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                num_classes=self.num_classes,
                padding_idx=self.padding_idx,
                device=self.device,
                dtype=self.dtype,
            ),
            ignore_lst=(None,),
        )


class IndexToName(Generic[T_Name], nn.Module):
    """
    For more information, see :func:`~torchwrench.nn.functional.multiclass.index_to_name`.
    """

    def __init__(
        self,
        idx_to_name: Union[Mapping[int, T_Name], Sequence[T_Name]],
    ) -> None:
        super().__init__()
        self.idx_to_name = idx_to_name

    def forward(
        self,
        index: Union[List[int], Tensor],
    ) -> List[T_Name]:
        name = index_to_name(index, self.idx_to_name)
        return name


class OnehotToIndex(Module):
    """
    For more information, see :func:`~torchwrench.nn.functional.multiclass.onehot_to_index`.
    """

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(
        self,
        onehot: Tensor,
    ) -> Tensor:
        index = onehot_to_index(onehot, dim=self.dim)
        return index

    def extra_repr(self) -> str:
        return dump_dict(dim=self.dim)


class OnehotToName(Generic[T_Name], nn.Module):
    """
    For more information, see :func:`~torchwrench.nn.functional.multiclass.onehot_to_name`.
    """

    def __init__(
        self,
        idx_to_name: Union[Mapping[int, T_Name], Sequence[T_Name]],
        dim: int = -1,
    ) -> None:
        super().__init__()
        self.idx_to_name = idx_to_name
        self.dim = dim

    def forward(
        self,
        onehot: Tensor,
    ) -> List[T_Name]:
        name = onehot_to_name(onehot, self.idx_to_name, dim=self.dim)
        return name

    def extra_repr(self) -> str:
        return dump_dict(dim=self.dim)


class NameToIndex(Generic[T_Name], nn.Module):
    """
    For more information, see :func:`~torchwrench.nn.functional.multiclass.name_to_index`.
    """

    def __init__(
        self,
        idx_to_name: Union[Mapping[int, T_Name], Sequence[T_Name]],
    ) -> None:
        super().__init__()
        self.idx_to_name = idx_to_name

    def forward(
        self,
        name: List[T_Name],
    ) -> Tensor:
        index = name_to_index(name, self.idx_to_name)
        return index


class NameToOnehot(Generic[T_Name], nn.Module):
    """
    For more information, see :func:`~torchwrench.nn.functional.multiclass.name_to_onehot`.
    """

    def __init__(
        self,
        idx_to_name: Union[Mapping[int, T_Name], Sequence[T_Name]],
        *,
        device: DeviceLike = None,
        dtype: DTypeLike = torch.bool,
    ) -> None:
        super().__init__()
        self.idx_to_name = idx_to_name
        self.device = device
        self.dtype = dtype

    def forward(
        self,
        name: List[T_Name],
    ) -> Tensor:
        onehot = name_to_onehot(
            name,
            self.idx_to_name,
            device=self.device,
            dtype=self.dtype,
        )
        return onehot

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                device=self.device,
                dtype=self.dtype,
            ),
            ignore_lst=(None,),
        )


class ProbsToIndex(Module):
    """
    For more information, see :func:`~torchwrench.nn.functional.multiclass.probs_to_index`.
    """

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(
        self,
        probs: Tensor,
    ) -> Tensor:
        index = probs_to_index(probs, dim=self.dim)
        return index

    def extra_repr(self) -> str:
        return dump_dict(dim=self.dim)


class ProbsToOnehot(Module):
    """
    For more information, see :func:`~torchwrench.nn.functional.multiclass.probs_to_onehot`.
    """

    def __init__(
        self,
        *,
        dim: int = -1,
        device: DeviceLike = None,
        dtype: DTypeLike = torch.bool,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.device = device
        self.dtype = dtype

    def forward(
        self,
        probs: Tensor,
    ) -> Tensor:
        onehot = probs_to_onehot(
            probs,
            dim=self.dim,
            device=self.device,
            dtype=self.dtype,
        )
        return onehot

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                dim=self.dim,
                device=self.device,
                dtype=self.dtype,
            ),
            ignore_lst=(None,),
        )


class ProbsToName(Generic[T_Name], nn.Module):
    """
    For more information, see :func:`~torchwrench.nn.functional.multiclass.probs_to_name`.
    """

    def __init__(
        self,
        idx_to_name: Union[Mapping[int, T_Name], Sequence[T_Name]],
        dim: int = -1,
    ) -> None:
        super().__init__()
        self.idx_to_name = idx_to_name
        self.dim = dim

    def forward(
        self,
        probs: Tensor,
    ) -> List[T_Name]:
        name = probs_to_name(probs, self.idx_to_name, dim=self.dim)
        return name

    def extra_repr(self) -> str:
        return dump_dict(dim=self.dim)
