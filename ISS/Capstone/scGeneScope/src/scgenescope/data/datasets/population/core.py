from __future__ import annotations

from collections.abc import Callable, Sequence
import logging
from typing import TypeVar
from dataclasses import dataclass
from pathlib import Path

import more_itertools
import numpy as np

from .utils import flatten_args
from ..samples.core import Samples
from ....types import Index, SupportsGrouping

logger = logging.getLogger(__name__)

T_indx = TypeVar("T", bound=[int, str])


@dataclass
class Population:

    samples: SupportsGrouping
    condition_on: str
    indexing: Index = Index.LOC
    transform: Callable | None = None
    include_condition: bool = False

    def __post_init__(self):
        self.indexing = Index(self.indexing)
        self.grouped = self.samples.groupby(self.condition_on)
        self.conditions = list(self.grouped.groups.keys())

    @property
    def indices(self):
        if self.indexing == Index.CONDITION:
            return self.conditions
        elif self.indexing == Index.LOC:
            return list(range(len(self)))

    def __len__(self):
        return len(self.conditions)

    def _get_condition(self, index: T_indx) -> str:
        if self.indexing == Index.LOC:
            try:
                condition = self.conditions[index]
            except TypeError as exc:
                raise TypeError(
                    f"Indexing mode is incompatible with index: {self.indexing=}, {index=}"
                ) from exc
        elif self.indexing == Index.CONDITION:
            condition = index
        else:
            raise RuntimeError(f"Invalid indexing mode: {self.indexing}")
        return condition

    def _fetch_indices(
        self, conditions: str | Sequence[str], squeezed: bool = True
    ) -> list[int] | list[list[int]]:
        """Get the indices of the samples corresponding to the condition.

        Args:
            conditions: the condition(s) to get the indices for. This can be a single
              a single string or a list of strings to support both __getexample__ and
              __getbatch__ methods, respectively.
            squeezed: if True return only a list of indices for single condition
                queries. If False, return a list of lists of indices for any query.

            whether to return a single list of indices or a list of lists
              of indices for the case with one condition.

        Returns:
            If the condition is a single string, return a list of indices for the
            samples corresponding to the condition. If the condition is a list of
            strings, return  a list of lists of indices for each condition in the
            list.
        """
        indices = []
        for condition in more_itertools.always_iterable(conditions):
            if condition in self.grouped.indices:
                indices.append(self.grouped.indices[condition])
            else:
                raise KeyError(
                    f"Condition '{condition}' not found in the grouped samples."
                )
        return indices[0] if squeezed and len(indices) == 1 else indices

    def __getexample__(self, index: T_indx):
        condition = self._get_condition(index)
        indices = self._fetch_indices(condition)
        example = self.samples.__getitem__(indices)
        # if include_condition is True, include the condition with the example
        if self.include_condition:
            try:
                example: dict
                example.update({"condition": condition})
            except AttributeError:
                example: tuple
                example = (example, condition)
        return example

    def __getitem__(self, index: T_indx):
        example = self.__getexample__(index)
        if self.transform is not None:
            example = self.transform(example)
        return example

    def __getbatch__(self, indices: list[T_indx]):
        try:
            conditions = [self._get_condition(index) for index in indices]
        except TypeError as exc:
            raise TypeError(
                f"Some indices are not compatible with indexing mode: {indices}"
            ) from exc
        index_lists = self._fetch_indices(conditions, squeezed=False)
        batch = flatten_args(self.samples.__getitem__)(*index_lists)
        # if include_condition is True, include the conditions with the batch
        if self.include_condition:
            batch: list[tuple]
            batch = (batch, conditions)

        return batch

    def __getitems__(self, indices: list[T_indx]):
        batch = self.__getbatch__(indices)
        if self.transform is not None:
            batch = self.transform(batch)
        return batch

    def __repr__(self):
        return f"{len(self)} populations conditioned by {self.condition_on} from: \n* {repr(self.samples)}"

    @classmethod
    def auto_factory(
        cls,
        file_path: Path | str,
        condition_on: str,
        indexing: Index = Index.LOC,
        include_condition: bool = False,
        **kwargs,
    ):
        samples = Samples.auto_factory(file_path, **kwargs)
        dataset = cls(
            samples, condition_on, indexing, include_condition=include_condition
        )
        return dataset

    @classmethod
    def from_factory(
        cls,
        samples_factory: Callable[[], SupportsGrouping],
        **kwargs,
    ):
        samples = samples_factory()
        dataset = cls(samples, **kwargs)
        return dataset
