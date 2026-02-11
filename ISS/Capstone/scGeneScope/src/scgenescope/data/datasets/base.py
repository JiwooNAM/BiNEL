from dataclasses import dataclass, KW_ONLY
from typing import Any, Callable, Sequence, TypeVar, Generic

import pandas as pd
import torch
from numpy.typing import ArrayLike

from ...types import Indexed

T = TypeVar("T", bound=ArrayLike)


@dataclass
class DataStore(torch.utils.data.Dataset, Generic[T]):
    """A Single-cell dataset representing a single modality of data.

    This dataset is a map-style dataset that can be indexed by an integer index.

    Each example is given by either:
      (1) an array-like object if no observations are available,
      (2) a tuple containing an array-like object and a pandas Series containing
        observations about the example if observations are available.

    Attributes:
        data_store: a store containing single data examples.
        observations: a pandas DataFrame containing the observations about each
          examples. Each column in the DataFrame corresponds to a different
          observation which we refer to as an axis.
    """

    store: Indexed[T]
    observations: pd.DataFrame | None
    _: KW_ONLY
    transform: Callable | None = None
    include_observations: bool = True

    def __post_init__(self):
        # verify all data examples have the same dimensionality
        if any(len(example) != self.dim for example in self.store):
            raise ValueError("All data examples must have the same dimensionality.")

        # Remove __getitems__ if store doesn't support list indexing
        # Note: It is important to do this here to ensure proper operation of
        # the torch.utils.data.DataLoader.
        try:
            _ = self.store[[0]]
        except (TypeError, IndexError):
            delattr(self, "__getitems__")

    @property
    def dim(self) -> int:
        """Return the dimensionality of the data examples."""
        return len(self.store[0])

    def __len__(self) -> int:
        return len(self.store)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the dataset (if viewed as array)."""
        return (len(self.store), self.dim)

    def __getexample__(self, indx: int) -> T | tuple[T, pd.Series]:
        example = self.store[indx]
        if self.include_observations and self.observations is not None:
            example = (example, self.observations.iloc[indx])
        return example

    def __getitem__(self, indx: int) -> Any:
        example = self.__getexample__(indx)
        if self.transform is not None:
            example = self.transform(example)
        return example

    def __getbatch__(
        self, indices: list[int]
    ) -> Sequence[T] | tuple[Sequence[T], pd.DataFrame]:
        """Get multiple items from the dataset."""
        try:
            batch = self.store[indices]
        except (TypeError, IndexError) as exc:
            raise TypeError("Dataset does not support batch indexing. ") from exc
        if self.include_observations and self.observations is not None:
            obs_batch = self.observations.iloc[indices]
            batch = (batch, obs_batch)
        return batch

    def __getitems__(self, indices: list[int]) -> list[Any]:
        """Get multiple items from the dataset."""
        batch = self.__getbatch__(indices)
        if self.transform is not None:
            batch = self.transform(batch)
        return batch

    def __repr__(self) -> str:
        ds_length = len(self.store)
        try:
            # Try to get shape and dtype directly
            dim = self.store[0].shape[0]
            dtype = self.store[0].dtype
        except (AttributeError, TypeError):
            try:
                # If that fails, try to handle as a sequence of items
                dim = ",".join(str(item.shape[0]) for item in self.store[0])
                dtype = ",".join(str(item.dtype) for item in self.store[0])
            except (AttributeError, TypeError):
                dim = None
                dtype = None
        obs_columns = self.observations.columns.values.tolist()
        cls_name = self.__class__.__name__
        as_str = f"{cls_name}({ds_length},[{dim}], <{dtype}>) with observations: {obs_columns}"
        return as_str
