from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import torch
from scipy.sparse import csr_matrix

from .core import ExampleTransform

T = TypeVar("T", np.ndarray, csr_matrix, torch.Tensor)


class Sampler(ExampleTransform):
    """
    Sample a random number of (k-1)-dim array from a k-dim array.

    Attributes:
        num_samples: range or number of entries to sample.
        seed: the seed for the random number generator.
    """

    def __init__(
        self,
        num_samples: int | tuple[int, int],
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.num_samples = num_samples
        if rng is not None:
            if seed is not None:
                raise ValueError("Cannot specify both `seed` and `rng`.")
            self._rng = rng
        else:
            self._rng = np.random.default_rng(seed)

    def __call__(
        self,
        data: T | Sequence[T],
        force_num_samples: int | None = None,
        generator: np.random.Generator | None = None,
    ) -> T | Sequence[T]:

        if generator is not None:
            rng = generator
        else:
            rng = self._rng

        if force_num_samples is None:
            try:
                low_bound, high_bound = self.num_samples
                num_samples = rng.integers(low_bound, high_bound)
            except TypeError:
                num_samples = self.num_samples
        else:
            num_samples = force_num_samples

        if isinstance(data, (list, tuple)):
            sampled_indices = rng.integers(data[0].shape[0], size=num_samples)
            sampled = []
            for datum in data:
                if datum.shape[0] < num_samples:
                    raise ValueError(
                        f"Number of samples {num_samples} is greater than the "
                        f"number of control cells {datum.shape[0]}."
                    )
                sampled.append(datum[sampled_indices])
        elif isinstance(data, dict):
            sampled_indices = rng.integers(
                next(iter(data.values())).shape[0], size=num_samples
            )
            for key, datum in data.items():
                sampled = {}
                if datum.shape[0] < num_samples:
                    raise ValueError(
                        f"Number of samples {num_samples} is greater than the "
                        f"number of control cells {datum.shape[0]}."
                    )
                sampled[key] = datum[sampled_indices]
        else:
            sampled_indices = rng.integers(data.shape[0], size=num_samples)
            if num_samples > data.shape[0]:
                raise ValueError(
                    f"Number of samples {self.num_samples} is greater than the "
                    f"number of control cells {data.shape[0]}."
                )
            sampled = data[sampled_indices]
        return sampled

    def __repr__(self):
        _base = super().__repr__()
        return _base.format(f"num_samples={self.num_samples}")
