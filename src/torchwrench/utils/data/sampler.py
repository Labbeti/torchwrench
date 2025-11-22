#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import random
from typing import Iterable, Iterator, List, Optional, Sequence, Union

import torch
from torch import Tensor
from torch.utils.data.sampler import Sampler

from torchwrench.nn.functional.transform import GeneratorLike, as_generator, as_tensor


class SubsetSampler(Sampler):
    """
    A sampler to loader a subset of a dataset from indices.
    """

    def __init__(self, indices: Union[List[int], Tensor]) -> None:
        if not isinstance(indices, Tensor):
            indices = as_tensor(indices)

        super().__init__()
        self._indices = indices

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices.tolist())

    def __len__(self) -> int:
        return len(self._indices)


class SubsetCycleSampler(Sampler):
    def __init__(
        self,
        indices: Union[Iterable[int], Tensor],
        n_max_iterations: Optional[int] = None,
        shuffle: bool = True,
        seed: GeneratorLike = None,
    ) -> None:
        """SubsetRandomSampler that cycle on indexes until a number max of iterations is reached.

        :param indexes: The list of indexes of the items.
        :param n_max_iterations: The maximal number of iterations. If None, it will be set to the length of indexes
                and the sampler will have the same behaviour than a SubsetRandomSampler.
                (default: None)
        :param shuffle: If True, shuffle the indexes at every len(indexes).
                (default: True)
        """
        if not isinstance(indices, Tensor):
            indices = as_tensor(indices, dtype="long")

        if n_max_iterations is None:
            n_max_iterations = len(indices)
        generator = as_generator(seed)

        super().__init__()
        self._indices = indices
        self._n_max_iterations = n_max_iterations
        self._shuffle = shuffle
        self._generator = generator

        self._shuffle_indices()

    def __iter__(self) -> Iterator[int]:
        for i, idx in enumerate(itertools.cycle(self._indices)):
            if i % len(self._indices) == len(self._indices) - 1:
                self._shuffle_indices()

            if i >= self._n_max_iterations:
                break

            yield idx.item()  # type: ignore

    def __len__(self) -> int:
        return self._n_max_iterations

    def _shuffle_indices(self):
        if self._shuffle:
            perm = torch.randperm(
                len(self._indices),
                device=self._indices.device,
                generator=self._generator,
            )
            self._indices = self._indices[perm]


class SubsetInfiniteCycleSampler(Sampler):
    """
    SubsetRandomSampler that cycle indefinitely on indexes.
    """

    def __init__(self, indexes: List[int], shuffle: bool = True) -> None:
        super().__init__()
        self.indexes = indexes
        self.shuffle = shuffle
        self._shuffle()

    def __iter__(self) -> Iterator[int]:
        for i, idx in enumerate(itertools.cycle(self.indexes)):
            if i % len(self.indexes) == len(self.indexes) - 1:
                self._shuffle()

            yield idx

    def __len__(self) -> int:
        raise NotImplementedError("Infinite sampler does not have __len__() method.")

    def _shuffle(self):
        if self.shuffle:
            random.shuffle(self.indexes)


class BalancedSampler(Sampler):
    def __init__(
        self,
        indexes_per_class: Sequence[Sequence[int]],
        n_max_samples: int,
        shuffle: bool = True,
    ) -> None:
        for cls_idx, indexes in enumerate(indexes_per_class):
            if len(indexes) == 0:
                msg = f"Found a class index {cls_idx} without any indexes."
                raise RuntimeError(msg)

        super().__init__()
        self.indexes_per_class = indexes_per_class
        self.n_max_samples = n_max_samples
        self.shuffle = shuffle

        self.max_idx = max(len(indexes) for indexes in self.indexes_per_class)
        self.pointers_per_class = [
            list(range(len(indexes))) for indexes in self.indexes_per_class
        ]
        self.local_idx_per_class = [0 for _ in range(len(self.indexes_per_class))]

        self.shuffle_pointers()

    def __iter__(self) -> Iterator[int]:
        global_idx = 0
        n_classes = len(self.indexes_per_class)
        for cls_idx in itertools.cycle(range(n_classes)):
            cls_idx: int

            if global_idx % self.max_idx == self.max_idx - 1:
                self.shuffle_pointers()

            class_indexes = self.indexes_per_class[cls_idx]
            pointers = self.pointers_per_class[cls_idx]
            pointer_idx = self.local_idx_per_class[cls_idx]

            pointer = pointers[pointer_idx]
            sample_idx = class_indexes[pointer]

            yield sample_idx

            self.local_idx_per_class[cls_idx] = (
                self.local_idx_per_class[cls_idx] + 1
            ) % len(pointers)
            global_idx += 1

    def __len__(self) -> int:
        return self.n_max_samples

    def shuffle_pointers(self):
        if self.shuffle:
            for pointers in self.pointers_per_class:
                random.shuffle(pointers)
