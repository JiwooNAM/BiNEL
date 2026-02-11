from dataclasses import dataclass
from collections.abc import Callable, Sequence

from .core import Population
from ..mixin import RandomStateMixin
from ...transforms.samplers import Sampler


@dataclass(kw_only=True)
class SampledPopulation(Population, RandomStateMixin):

    num_samples: int | tuple[int, int] | None = None
    sampler: Callable[[list[int]], list[int]] | None = None
    seed: int | None = None

    def __post_init__(self):
        super().__post_init__()
        self.set_state(seed=self.seed)
        if self.sampler is None:
            if self.num_samples:
                self.sampler = Sampler(self.num_samples, rng=self.state)
            else:
                raise ValueError("Sampler or number of samples must be provided.")

    def _fetch_indices(
        self, conditions: str | Sequence[str], squeezed: bool = True
    ) -> list[int] | list[list[int]]:
        """Fetch and sample the indices of the samples."""
        index_lists = super()._fetch_indices(conditions, squeezed=False)
        sampled_indices = [
            self.sampler(group, generator=self.state) for group in index_lists
        ]
        return (
            sampled_indices[0]
            if squeezed and len(sampled_indices) == 1
            else sampled_indices
        )
