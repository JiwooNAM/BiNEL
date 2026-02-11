from dataclasses import dataclass, KW_ONLY
from typing import Callable, Any, Sequence

import more_itertools
import numpy as np
from pandas.core.groupby.generic import DataFrameGroupBy


@dataclass
class ContextMixin:
    _: KW_ONLY
    context_params: list[str] | None = None
    compute_context: Callable | None = None

    def fetch_context_params(self) -> dict[str, Any]:
        params = {param: getattr(self, param, None) for param in self.context_params}
        return params

    @property
    def context(self) -> dict[str, Any]:
        if self.context_params and self.compute_context:
            params = self.fetch_context_params()
            params = self.compute_context(params)
        else:
            params = {}
        return params


class ObservationGroupingMixin:

    def groupby(self, axis: str) -> DataFrameGroupBy:
        """Group the dataset by a given axis.

        Args:
            axis: the axis to group the dataset by. For example, "condition"
              groups the dataset by the condition axis.

        Returns:
            A pandas DataFrameGroupBy object that groups the dataset along the
            specified axis.
        """
        try:
            grouped = self.observations.groupby(axis, observed=True)
        except KeyError as exc:
            raise ValueError(f"Invalid axis: {axis}") from exc
        except AttributeError as exc:
            raise ValueError(
                f"{self} has no observations available for grouping. "
                f"Ensure the dataset has an `observations` attribute."
            ) from exc

        return grouped


class RandomStateMixin:
    """A mixin class for handling numpy random state."""

    @property
    def state(self):
        """The random number generator to use for randomness.

        Returns:
            A numpy random number generator.
        """
        if self._states is None:
            self.set_state(None)
        assert self._states is not None
        return next(self._states)

    @property
    def rng(self):
        return self.state

    def set_state(
        self,
        rngs: np.random.Generator | Sequence[np.random.Generator] | None = None,
        seed: int | None = None,
    ):
        """Set the random number generator.

        Args:
            rngs: the random number generator to use for randomness. If Sequence,
              then the random number generator is cycled through the sequence
              for each iteration. Once the sequence is exhausted, it remains at
              the last generator. If None, the default numpy random number
              generator is used.
            seed: the seed to use to create a random number generator. If None,
              the default numpy random number generator is used.

        Raises:
            ValueError: if both `rngs` and `seed` are specified
        """
        if rngs is not None and seed is not None:
            raise ValueError("Cannot specify both `seed` and `rngs`.")

        if rngs is None:
            rngs = np.random.default_rng(seed)

        self._states = iter(
            more_itertools.repeat_last(more_itertools.always_iterable(rngs))
        )

    @state.setter
    def state(self, rngs: np.random.Generator | Sequence[np.random.Generator] | None):
        self.set_state(rngs)
