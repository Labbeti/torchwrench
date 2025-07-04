#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from pythonwrench.collections import dump_dict
from pythonwrench.typing import BuiltinScalar, SupportsIterLen
from torch import Tensor, nn

from torchwrench.core.make import DeviceLike, DTypeLike, GeneratorLike
from torchwrench.extras.numpy import np
from torchwrench.nn.functional.transform import (
    PadCropAlign,
    PadMode,
    PadValue,
    SqueezeMode,
    T_BuiltinScalar,
    as_tensor,
    flatten,
    identity,
    move_to_rec,
    pad_and_crop_dim,
    repeat_interleave_nd,
    resample_nearest_freqs,
    resample_nearest_rates,
    resample_nearest_steps,
    shuffled,
    squeeze,
    to_item,
    top_p,
    topk,
    transform_drop,
    unsqueeze,
    view_as_complex,
    view_as_real,
)
from torchwrench.types._typing import (
    ComplexFloatingTensor,
    LongTensor,
    ScalarLike,
    T_TensorOrArray,
)
from torchwrench.utils import return_types

from .module import EModule, Module

T = TypeVar("T")


class AsTensor(Module):
    """
    Module version of :func:`~to.as_tensor`.
    """

    def __init__(
        self,
        *,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, x: Any) -> Tensor:
        return as_tensor(x, dtype=self.dtype, device=self.device)

    def extra_repr(self) -> str:
        return dump_dict(
            dtype=self.dtype,
            device=self.device,
            ignore_lst=(None,),
        )


class Flatten(Module):
    def __init__(self, start_dim: int = 0, end_dim: Optional[int] = None) -> None:
        """
        For more information, see :func:`~torchwrench.nn.functional.transform.flatten`.
        """
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    @overload
    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        ...

    @overload
    def forward(self, x: Union[np.ndarray, np.generic]) -> np.ndarray: ...

    @overload
    def forward(self, x: T_BuiltinScalar) -> List[T_BuiltinScalar]: ...

    @overload
    def forward(self, x: Iterable[T_BuiltinScalar]) -> List[T_BuiltinScalar]:  # type: ignore
        ...

    def forward(self, x: Any) -> Any:
        return flatten(
            x,
            start_dim=self.start_dim,
            end_dim=self.end_dim,
        )

    def extra_repr(self) -> str:
        return dump_dict(
            start_dim=self.start_dim,
            end_dim=self.end_dim,
        )


class Identity(Module):
    def __init__(self, *args, **kwargs) -> None:
        """Identity class placeholder.

        Unlike torch.nn.Identity which only supports Tensor typing, its type output is the same than its input type.
        """
        super().__init__()

    def forward(self, x: T) -> T:
        return identity(x)


class MoveToRec(Module):
    """
    Module version of :func:`~torchwrench.move_to_rec`.
    """

    def __init__(
        self,
        predicate: Optional[Callable[[Union[Tensor, nn.Module]], bool]] = None,
    ) -> None:
        super().__init__()
        self.predicate = predicate

    def forward(self, x: Any) -> Any:
        return move_to_rec(x, predicate=self.predicate)


class PadAndCropDim(Module):
    def __init__(
        self,
        target_length: int,
        align: PadCropAlign = "left",
        pad_value: PadValue = 0.0,
        dim: int = -1,
        mode: PadMode = "constant",
        generator: GeneratorLike = None,
    ) -> None:
        """
        For more information, see :func:`~torchwrench.nn.functional.transform.pad_and_crop_dim`.
        """
        super().__init__()
        self.target_length = target_length
        self.align: PadCropAlign = align
        self.pad_value = pad_value
        self.dim = dim
        self.mode: PadMode = mode
        self.generator: GeneratorLike = generator

    def forward(self, x: Tensor) -> Tensor:
        return pad_and_crop_dim(
            x,
            self.target_length,
            align=self.align,
            pad_value=self.pad_value,
            dim=self.dim,
            mode=self.mode,
            generator=self.generator,
        )

    def extra_repr(self) -> str:
        return dump_dict(
            target_length=self.target_length,
            align=self.align,
            pad_value=self.pad_value,
            dim=self.dim,
            mode=self.mode,
        )


class RepeatInterleaveNd(Module):
    """
    For more information, see :func:`~to.repeat_interleave_nd`.
    """

    def __init__(self, repeats: int, dim: int) -> None:
        super().__init__()
        self.repeats = repeats
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return repeat_interleave_nd(x, self.repeats, self.dim)

    def extra_repr(self) -> str:
        return dump_dict(repeats=self.repeats, dim=self.dim)


class ResampleNearestRates(Module):
    """
    For more information, see :func:`~torchwrench.nn.functional.transform.resample_nearest_rates`.
    """

    def __init__(
        self,
        rates: Union[float, Iterable[float]],
        dims: Union[int, Iterable[int]] = -1,
        round_fn: Callable[[Tensor], Tensor] = torch.floor,
    ) -> None:
        super().__init__()
        self.rates = rates
        self.dims = dims
        self.round_fn = round_fn

    def forward(self, x: Tensor) -> Tensor:
        return resample_nearest_rates(
            x,
            rates=self.rates,
            dims=self.dims,
            round_fn=self.round_fn,
        )

    def extra_repr(self) -> str:
        return dump_dict(rates=self.rates, dims=self.dims)


class ResampleNearestFreqs(Module):
    def __init__(
        self,
        orig_freq: int,
        new_freq: int,
        dims: Union[int, Iterable[int]] = -1,
        round_fn: Callable[[Tensor], Tensor] = torch.floor,
    ) -> None:
        """
        For more information, see :func:`~torchwrench.nn.functional.transform.resample_nearest_freqs`.
        """
        super().__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.dims = dims
        self.round_fn = round_fn

    def forward(self, x: Tensor) -> Tensor:
        return resample_nearest_freqs(
            x,
            orig_freq=self.orig_freq,
            new_freq=self.new_freq,
            dims=self.dims,
            round_fn=self.round_fn,
        )

    def extra_repr(self) -> str:
        return dump_dict(
            orig_freq=self.orig_freq, new_freq=self.new_freq, dims=self.dims
        )


class ResampleNearestSteps(Module):
    def __init__(
        self,
        steps: Union[float, Iterable[float]],
        dims: Union[int, Iterable[int]] = -1,
        round_fn: Callable[[Tensor], Tensor] = torch.floor,
    ) -> None:
        """
        For more information, see :func:`~torchwrench.nn.functional.transform.resample_nearest_steps`.
        """
        super().__init__()
        self.steps = steps
        self.dims = dims
        self.round_fn = round_fn

    def forward(self, x: Tensor) -> Tensor:
        return resample_nearest_steps(
            x,
            steps=self.steps,
            dims=self.dims,
            round_fn=self.round_fn,
        )

    def extra_repr(self) -> str:
        return dump_dict(steps=self.steps, dims=self.dims)


class Squeeze(Module):
    """
    Module version of :func:`~torchwrench.squeeze`.
    """

    def __init__(
        self,
        dim: Union[int, Iterable[int], None] = None,
        mode: SqueezeMode = "view_if_possible",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.mode: SqueezeMode = mode

    def forward(self, x: Tensor) -> Tensor:
        return squeeze(x, self.dim, self.mode)

    def extra_repr(self) -> str:
        return dump_dict(
            dim=self.dim,
            mode=self.mode,
        )


class Shuffled(Module):
    def __init__(
        self,
        dims: Union[int, Iterable[int]] = -1,
        generator: GeneratorLike = None,
    ) -> None:
        """
        For more information, see :func:`~torchwrench.nn.functional.transform.shuffled`.
        """
        super().__init__()
        self.dims = dims
        self.generator: GeneratorLike = generator

    def forward(self, x: Tensor) -> Tensor:
        return shuffled(x, dims=self.dims, generator=self.generator)

    def extra_repr(self) -> str:
        return dump_dict(dims=self.dims)


class ToItem(Module):
    """
    Module version of :func:`~torchwrench.to_item`.
    """

    def forward(
        self,
        x: Union[ScalarLike, Tensor, np.ndarray, SupportsIterLen],
    ) -> BuiltinScalar:
        return to_item(x)  # type: ignore


class TransformDrop(Generic[T], EModule[T, T]):
    def __init__(
        self,
        transform: Callable[[T], T],
        p: float,
        generator: GeneratorLike = None,
    ) -> None:
        """
        For more information, see :func:`~torchwrench.nn.functional.transform.transform_drop`.
        """
        super().__init__()
        self.transform = transform
        self.p = p
        self.generator: GeneratorLike = generator

    def forward(self, x: T) -> T:
        return transform_drop(
            transform=self.transform,
            x=x,
            p=self.p,
            generator=self.generator,
        )

    def extra_repr(self) -> str:
        return dump_dict(p=self.p)


class Unsqueeze(Module):
    """
    Module version of :func:`~torchwrench.unsqueeze`.
    """

    def __init__(
        self,
        dim: Union[int, Iterable[int]],
        mode: SqueezeMode = "view_if_possible",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.mode: SqueezeMode = mode

    def forward(self, x: T_TensorOrArray) -> T_TensorOrArray:
        return unsqueeze(x, self.dim, self.mode)

    def extra_repr(self) -> str:
        return dump_dict(dim=self.dim, mode=self.mode)


class ViewAsReal(Module):
    """
    Module version of :func:`~torchwrench.to_item`.
    """

    def forward(
        self, x: Union[Tensor, np.ndarray, complex]
    ) -> Union[Tensor, np.ndarray, Tuple[float, float]]:
        return view_as_real(x)


class ViewAsComplex(Module):
    """
    Module version of :func:`~torchwrench.to_item`.
    """

    def forward(
        self, x: Union[Tensor, np.ndarray, Tuple[float, float]]
    ) -> Union[ComplexFloatingTensor, np.ndarray, complex]:
        return view_as_complex(x)


class Topk(Module):
    """
    Module version of :func:`~torchwrench.topk`.
    """

    def __init__(
        self,
        k: int,
        dim: int = -1,
        largest: bool = True,
        sorted: bool = True,
        *,
        return_values: bool = True,
        return_indices: bool = True,
    ) -> None:
        if not return_values and not return_indices:
            msg = f"Invalid combinaison of arguments {return_values=} and {return_indices=}. (at least one of them must be True)"
            raise ValueError(msg)

        super().__init__()
        self.k = k
        self.dim = dim
        self.largest = largest
        self.sorted = sorted
        self.return_values = return_values
        self.return_indices = return_indices

    def forward(self, x: Tensor) -> Union[Tensor, LongTensor, return_types.topk]:
        return topk(
            x=x,
            k=self.k,
            dim=self.dim,
            largest=self.largest,
            sorted=self.sorted,
            return_values=self.return_values,
            return_indices=self.return_indices,
        )

    def extra_repr(self) -> str:
        return dump_dict(
            k=self.k,
            dim=self.dim,
            largest=self.largest,
            sorted=self.sorted,
            return_values=self.return_values,
            return_indices=self.return_indices,
        )


class TopP(Module):
    """
    Module version of :func:`~torchwrench.top_p`.
    """

    def __init__(
        self,
        p: float,
        dim: int = -1,
        largest: bool = True,
        *,
        return_values: bool = True,
        return_indices: bool = True,
    ) -> None:
        if not return_values and not return_indices:
            msg = f"Invalid combinaison of arguments {return_values=} and {return_indices=}. (at least one of them must be True)"
            raise ValueError(msg)

        super().__init__()
        self.p = p
        self.dim = dim
        self.largest = largest
        self.return_values = return_values
        self.return_indices = return_indices

    def forward(self, x: Tensor) -> Union[Tensor, LongTensor, return_types.top_p]:
        return top_p(
            x=x,
            p=self.p,
            dim=self.dim,
            largest=self.largest,
            return_values=self.return_values,
            return_indices=self.return_indices,
        )

    def extra_repr(self) -> str:
        return dump_dict(
            p=self.p,
            dim=self.dim,
            largest=self.largest,
            return_values=self.return_values,
            return_indices=self.return_indices,
        )


# Aliases
ToTensor = AsTensor
TopK = Topk
