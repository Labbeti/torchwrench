#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
from typing import Iterable, Iterator, List, Literal, Sequence, Union

from torch import Tensor
from torch.utils.data.sampler import Sampler

from torchwrench.nn.functional.transform import (
    GeneratorLike,
    as_generator,
    as_tensor,
    shuffled,
)


class SubsetSampler(Sampler[int]):
    def __init__(self, indices: Union[List[int], Tensor]) -> None:
        """
        A sampler to load a subset of a dataset from indices.

        Args:
            indices: List of indices to return.
        """
        indices = as_tensor(indices, dtype="long")

        super().__init__(None)
        self._indices = indices

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices.tolist())

    def __len__(self) -> int:
        return len(self._indices)


class SubsetCycleSampler(Sampler[int]):
    def __init__(
        self,
        indices: Union[Tensor, Iterable[int]],
        n_max_iterations: Union[int, Literal["inf"]] = "inf",
        shuffle: bool = True,
        seed: GeneratorLike = None,
    ) -> None:
        """SubsetCycleSampler that cycle on indices indifinitely or until a number max of iterations is reached.

        Args:
            indices: The list of indices of the items.
            n_max_iterations: The maximal number of iterations.
                If "inf", any call to __len__ will raises a NotImplementedError exception.
                defaults to "inf".
            shuffle: If True, shuffle the indices at every len(indices).
                defaults to True.
            seed: Optional seed or generator used to shuffle indices.
                defaults to None.
        """
        indices = as_tensor(indices, dtype="long")
        generator = as_generator(seed)

        super().__init__(None)
        self._indices = indices
        self._n_max_iterations = n_max_iterations
        self._shuffle = shuffle
        self._generator = generator

        self._shuffle_indices()

    def __iter__(self) -> Iterator[int]:
        for i, idx in enumerate(itertools.cycle(self._indices)):
            if i % len(self._indices) == len(self._indices) - 1:
                self._shuffle_indices()

            if isinstance(self._n_max_iterations, int) and i >= self._n_max_iterations:
                break

            yield idx.item()  # type: ignore

    def __len__(self) -> int:
        if isinstance(self._n_max_iterations, int):
            return self._n_max_iterations
        elif self._n_max_iterations == "inf":
            msg = "Infinite sampler does not have __len__() method."
            raise NotImplementedError(msg)
        else:
            msg = f"Invalid argument {self._n_max_iterations=}."
            raise ValueError(msg)

    def _shuffle_indices(self) -> None:
        if not self._shuffle:
            return None
        self._indices = shuffled(self._indices, generator=self._generator)


class BalancedSampler(Sampler):
    def __init__(
        self,
        indices_per_class: Sequence[Sequence[int]],
        n_max_iterations: int,
        shuffle: bool = True,
        seed: GeneratorLike = None,
    ) -> None:
        """BalancedSampler class.

        Args:
            indices_per_class: List of indices per class index.
            n_max_iterations: The maximal number of iterations.
                If "inf", any call to __len__ will raises a NotImplementedError exception.
                defaults to "inf".
            shuffle: If True, shuffle the indices at every len(indices).
                defaults to True.
            seed: Optional seed or generator used to shuffle indices.
                defaults to None.
        """
        for cls_idx, indices in enumerate(indices_per_class):
            if len(indices) == 0:
                msg = f"Found a class index {cls_idx} without any indices."
                raise RuntimeError(msg)

        max_idx = max(len(indices) for indices in indices_per_class)
        pointers_per_class = [
            list(range(len(indices))) for indices in indices_per_class
        ]
        local_idx_per_class = [0 for _ in range(len(indices_per_class))]
        generator = as_generator(seed)

        super().__init__(None)
        self._indices_per_class = indices_per_class
        self._n_max_iterations = n_max_iterations
        self._shuffle = shuffle
        self._generator = generator

        self._max_idx = max_idx
        self._pointers_per_class = pointers_per_class
        self._local_idx_per_class = local_idx_per_class

        self._shuffle_indices()

    def __iter__(self) -> Iterator[int]:
        global_idx = 0
        n_classes = len(self._indices_per_class)
        for cls_idx in itertools.cycle(range(n_classes)):
            if global_idx >= self._n_max_iterations:
                break
            if global_idx % self._max_idx == self._max_idx - 1:
                self._shuffle_indices()

            class_indices = self._indices_per_class[cls_idx]
            pointers = self._pointers_per_class[cls_idx]
            pointer_idx = self._local_idx_per_class[cls_idx]

            pointer = pointers[pointer_idx]
            sample_idx = class_indices[pointer]

            yield sample_idx

            self._local_idx_per_class[cls_idx] = (pointer_idx + 1) % len(pointers)
            global_idx += 1

    def __len__(self) -> int:
        return self._n_max_iterations

    def _shuffle_indices(self) -> None:
        if not self._shuffle:
            return None

        for i, pointers in enumerate(self._pointers_per_class):
            pointers_pt = as_tensor(pointers)
            pointers_pt = shuffled(pointers_pt, generator=self._generator)
            self._pointers_per_class[i] = pointers_pt.tolist()
