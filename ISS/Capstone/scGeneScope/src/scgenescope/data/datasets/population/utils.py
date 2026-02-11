from collections.abc import Callable
import functools
import math
import warnings

import more_itertools
import numpy as np
import torch


def flatten_args(func: Callable) -> Callable:
    """Flattens the arguments of a function before calling it."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # call the function with the flattened arguments
        output = func(np.concatenate(args), **kwargs)
        # split the output based on the sizes of the input arguments
        lengths = [len(arg) for arg in args]
        splitter = functools.partial(
            np.split, indices_or_sections=np.cumsum(lengths)[:-1]
        )
        try:
            split_output = splitter(output)
        except ValueError:
            split_output = [*zip(*map(splitter, output))]
        return split_output

    return wrapper


class MakeIterable(torch.utils.data.IterableDataset):

    def __init__(
        self,
        dataset,
        num_populations: int | None = None,
        resample_factor: int = 1,
        seed: int | None = None,
        batch_size: int | None = None,
        collate_fn: Callable | None = None,
    ):
        self.dataset = dataset

        self.num_populations = num_populations
        self.resample_factor = resample_factor
        self.seed = seed

        self.batch_size = batch_size
        self.collate_fn = collate_fn

        if self.num_populations is None:
            self.num_populations = len(self.dataset)

        self._rng = torch.Generator()
        if self.seed is not None:
            self._rng.manual_seed(self.seed)

        if self.num_populations > (num_populations_in_data := len(self.dataset)):
            try:
                self.populations = [
                    self.dataset.indices[
                        torch.randint(0, num_populations_in_data, (1,))
                    ]
                    for _ in range(self.num_populations)
                ]
            except AttributeError:
                self.populations = [
                    self.conditions[torch.randint(0, num_populations_in_data, (1,))]
                    for _ in range(self.num_populations)
                ]
        else:
            try:
                self.populations = self.dataset.indices
            except AttributeError:
                self.populations = self.dataset.conditions

        self._samples_seeds_per_condition = torch.randint(
            0, 2**32, (self.num_populations,), generator=self._rng
        )

    def set_transform(self, transform: Callable):
        """Re-assigns the transform to the dataset."""
        self.dataset.transform = transform

    @property
    def transform(self) -> Callable:
        return self.dataset.transform

    @transform.setter
    def transform(self, transform: Callable):
        self.dataset.transform = transform

    @property
    def batched(self) -> bool:
        return bool(self.batch_size)

    def __len__(self) -> int:
        total_samples = self.num_populations * self.resample_factor
        return total_samples // self.batch_size if self.batched else total_samples

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # single-process data loading
            indices = self.populations
            seeds = self._samples_seeds_per_condition
        else:
            # in a worker process: split workload
            per_worker = int(
                math.ceil(len(self.populations) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.populations))
            indices = self.populations[iter_start:iter_end]
            seeds = self._samples_seeds_per_condition[iter_start:iter_end]

        # rngs = [torch.generator().manual_seed(seed.item()) for seed in seeds]
        rngs = [np.random.default_rng(seed.item()) for seed in seeds]

        if self.resample_factor > 1:
            indices = more_itertools.ncycles(indices, self.resample_factor)
            rngs = more_itertools.ncycles(rngs, self.resample_factor)

        if self.batched:
            yield from self.__genitems__(indices, rngs)
        else:
            yield from self.__genitem__(indices, rngs)

    def __genitem__(self, indices, rngs):
        for indx, rng in zip(indices, rngs):
            try:
                self.dataset.set_state(rng)
            except AttributeError:
                with warnings.catch_warnings(action="ignore"):
                    warnings.warn(
                        f"Dataset {self.dataset} does not support setting the random state. "
                        f"Continuing without setting the random state."
                    )
            example = self.dataset.__getitem__(indx)
            yield example

    def __genitems__(self, indices, rngs):
        # create batches of indices and rngs
        batched_indices = more_itertools.batched(indices, self.batch_size)
        batched_rngs = more_itertools.batched(rngs, self.batch_size)
        # iterate over the batches
        for indices_batch, rngs_batch in zip(batched_indices, batched_rngs):
            try:
                self.dataset.set_state(rngs_batch)
            except AttributeError:
                with warnings.catch_warnings(action="ignore"):
                    warnings.warn(
                        f"Dataset {self.dataset} does not support setting the random state. "
                        f"Continuing without setting the random state."
                    )
            batch = self.dataset.__getitems__(indices_batch)
            if self.collate_fn is not None:
                batch = self.collate_fn(batch)
            yield batch

    @classmethod
    def from_factory(
        cls,
        factory: Callable,
        **kwargs,
    ):
        dataset = factory()
        return cls(dataset, **kwargs)
