from __future__ import annotations

from collections.abc import Callable
import functools
import itertools
import logging
import operator
from typing import Sequence
import warnings

import numpy as np
import torch
from p_tqdm import p_map, t_map

from .core import Population
from ..mixin import RandomStateMixin
from ....types import Index, SupportsIndexModeSelect

logger = logging.getLogger(__name__)


class ConditionAlignedPopulations(torch.utils.data.StackDataset, RandomStateMixin):

    def __init__(
        self,
        *datasets: SupportsIndexModeSelect,
        transform: Callable | None = None,
        seed: int | None = None,
        **named_datasets: SupportsIndexModeSelect,
    ):
        # Check whether all datasets have the same conditions
        conditions = [
            set(dataset.conditions)
            for dataset in itertools.chain(datasets, named_datasets.values())
        ]
        alignment_conditions = set.intersection(*conditions)
        if any(condition != alignment_conditions for condition in conditions):
            raise ValueError("Datasets must have the same conditions.")

        # Make sure all datasets are condition-indexed
        if any(
            dataset.indexing is not Index.CONDITION
            for dataset in itertools.chain(datasets, named_datasets.values())
        ):
            raise ValueError("All datasets must be condition-indexed.")

        # Initialize StackDataset
        super().__init__(*datasets, **named_datasets)

        # Remove the __getitems__ method from the stack dataset because it has a
        # default fallback implementation that will collect examples from each
        # dataset in the stack. This is not the desired behavior for this class:
        # they claim __getitems__ should be used only if the constituent datasets
        # also support it.
        if any(
            not callable(getattr(dataset, "__getitems__", None))
            for dataset in itertools.chain(datasets, named_datasets.values())
        ):
            self.__getitems__ = None

        self.conditions = list(alignment_conditions)
        self.transform = transform
        self.set_state(seed=seed)

    @property
    def indices(self) -> list[int]:
        return list(range(len(self.conditions)))

    def __getexample__(self, index: int):
        condition = self.conditions[index]
        example = super().__getitem__(condition)
        try:
            example: tuple
            example += (condition,)
        except TypeError:
            example: dict
            example.update({"condition": condition})
        return example

    def set_state(
        self,
        rngs: np.random.Generator | Sequence[np.random.Generator] | None = None,
        seed: int | None = None,
    ):
        RandomStateMixin.set_state(self, rngs=rngs, seed=seed)
        try:
            self.datasets: dict[str, SupportsIndexModeSelect]
            for name, ds in self.datasets.items():
                try:
                    ds.set_state(rngs=rngs)
                except AttributeError:
                    raise RuntimeError
                    with warnings.catch_warnings(action="ignore"):
                        warnings.warn(f"{name} does not support set_state.")
        except AttributeError:
            self.datasets: list[SupportsIndexModeSelect]
            for ds in self.datasets:
                try:
                    ds.set_state(rngs=rngs)
                except AttributeError:
                    with warnings.catch_warnings(action="ignore"):
                        warnings.warn(f"{ds} does not support set_state.")

    def __getitem__(self, index: int):
        example = self.__getexample__(index)
        if self.transform is not None:
            example = self.transform(example)
        return example

    def __getbatch__(self, indices: list[int]):
        conditions = [self.conditions[index] for index in indices]
        batch = super().__getitems__(conditions)
        try:
            batch: list[dict]
            for example, condition in zip(batch, conditions, strict=True):
                example.update({"condition": condition})
        except AttributeError:
            batch: list[tuple]
            batch = [
                (*data, condition)
                for data, condition in zip(batch, conditions, strict=True)
            ]
        return batch

    def __getitems__(self, indices: list[int]):
        batch = self.__getbatch__(indices)
        if self.transform is not None:
            batch = self.transform(batch)
        return batch

    @classmethod
    def join_on_condition(
        cls,
        *dataset_factories: Callable[[], SupportsIndexModeSelect],
        condition: str,
        use_multiprocess_loading: bool = False,
        transform: Callable | None = None,
        **named_dataset_factories: Callable,
    ) -> ConditionAlignedPopulations:
        """Create and combine multiple datasets on a common condition.

        Args:
            dataset_factories: A sequence of callables that accepts `condition_on`
              and `indexing` and that return datasets.
            condition: The condition to align the datasets on.
            use_multiprocess_loading: Whether to use multiprocessing to load the datasets.
            transform: A callable that transforms the examples or batches.
            named_dataset_factories: A dictionary where each key-value pair contains the
              name of the dataset and a callable that accepts `condition_on` and
              `indexing` and that return datasets.

        Returns:
            A combined multimodal dataset consisting of the instantiated
            unimodal population datasets.
        """

        if dataset_factories:
            if named_dataset_factories:
                raise ValueError("Cannot mix ordered and named datasets specs.")
            # add condition and indexing=condition to each dataset factory. This is
            # is necessary because the datasets need to be condition indexed for the
            # conditional alignment to work.
            dataset_factories = [
                functools.partial(
                    factory, condition_on=condition, indexing=Index.CONDITION
                )
                for factory in dataset_factories
            ]
            # instantiate the datasets
            datasets = (
                p_map(operator.call, dataset_factories, num_cpus=len(dataset_factories))
                if use_multiprocess_loading
                else t_map(operator.call, dataset_factories)
            )
            named_datasets = {}
        elif named_dataset_factories:
            datasets = []
            # add condition and indexing=condition to each dataset factory
            named_dataset_factories = {
                name: functools.partial(
                    factory, condition_on=condition, indexing=Index.CONDITION
                )
                for name, factory in named_dataset_factories.items()
            }
            # instantiate the datasets
            named_datasets = {
                name: dataset
                for name, dataset in zip(
                    named_dataset_factories.keys(),
                    (
                        p_map(
                            operator.call,
                            named_dataset_factories.values(),
                            num_cpus=len(named_dataset_factories),
                        )
                        if use_multiprocess_loading
                        else t_map(operator.call, named_dataset_factories.values())
                    ),
                )
            }
        else:
            raise ValueError("No datasets specs provided.")
        # combine the datasets using the ConditionAlignedPopulations constructor
        combined_dataset = cls(*datasets, transform=transform, **named_datasets)
        return combined_dataset

    # alias for join_on_condition for backward compatibility (will be deprecated)
    combine_samples = join_on_condition
