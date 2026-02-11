from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence, Set, Iterator
from dataclasses import dataclass
from typing import Self

import torch
import torch_geometric
import torch_geometric.data


class BatchedSets(ABC):
    """A batch of sets."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """The dimensionality of the elements in the set."""

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """The number of examples in the batch."""

    @classmethod
    @abstractmethod
    def collate(cls, batch: Sequence[Set]) -> Self:
        """Collate a sequence of sets into a batch.

        Args:
            batch: A sequence of sets.

        Returns:
            A batch of sets.
        """

    @abstractmethod
    def disperse(self) -> Sequence[Set]:
        """Disperse the batch into a sequence of sets.

        Returns:
            A sequence of sets.
        """


@dataclass
class TensorSet(Set):
    """A set implemented as a tensor.

    The set is represented as a tensor where each row is an element in the set.
    A set S = {x_1, x_2, ..., x_N} is represented as a tensor of shape [N x dim]
    where N is the cardinality of the set and dim is the dimensionality of the
    elements.

    Attributes:
        tensor: A tensor of shape [N x dim] where N is the cardinality of the
          set and dim is the dimensionality of the elements.
    """

    tensor: torch.Tensor

    def __post_init__(self):
        if not isinstance(self.tensor, torch.Tensor):
            self.tensor = torch.tensor(self.tensor)
        if self.tensor.ndim != 2:
            raise ValueError("The tensor must have exactly two dimensions.")

    def __len__(self) -> int:
        return self.tensor.size(0)

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self.tensor)

    def __contains__(self, item: torch.Tensor) -> bool:
        return (item == self.tensor).all(dim=1).any().item()

    def to_pyg(self) -> PyGSet:
        """Convert the tensor set to a PyTorch Geometric set."""
        return PyGSet(self.tensor)

    @classmethod
    def new(cls, tensor: torch.Tensor, **kwargs) -> Self:
        return cls(torch.tensor(tensor, **kwargs))

    def __getattr__(self, name: str):
        """Delegate unimplemented methods to the tensor member."""
        return getattr(self.tensor, name)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        args = [getattr(a, "tensor", a) for a in args]
        ret = func(*args, **kwargs)
        return ret


class PyGSet(torch_geometric.data.Data, Set):
    """A set implemented as a PyTorch Geometric graph.

    The set is represented as an edgeless graph where each node is an element in
    the set. A set S = {x_1, x_2, ..., x_N} is represented as a graph with zero
    edges and N nodes where each node corresponds to an element in the set.

    Attributes:
        data: A PyTorch Geometric graph where each node corresponds to an element
          in the set.
    """

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        return cls(x=tensor)

    def __len__(self) -> int:
        return self.num_nodes

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self.x)

    def __contains__(self, item: torch.Tensor) -> bool:
        return (item == self.x).all(dim=1).any().item()

    def to_tensor(self) -> TensorSet:
        """Convert the PyTorch Geometric set to a tensor set."""
        return TensorSet(self.x)


@dataclass
class BatchedEqualSizedTensorSets(BatchedSets):
    """A batch of tensor sets of the same cardinality and dimensionality.

    The batch is represented as a tensor where the third dimension is the number
    of examples.

    A batch of sets S^i = {x_1^i, x_2^i, ..., x_N^i} for i=1, ... ,B is
    represented as a tensor of shape (batch_size, number_elements, dim) where
    number_elements is the cardinality of the sets, dim is the dimensionality
    of the elements, and batch_size is the number of sets in the batch.

    Attributes:
        tensor: A tensor of shape (batch_size, number_elements, dim) where
          batch_size is the number of sets in the batch, number_elements is the
          cardinality of the sets in the batch, and dim is the dimensionality
          of the elements.
    """

    tensor: torch.Tensor

    def __post_init__(self):
        if not isinstance(self.tensor, torch.Tensor):
            self.tensor = torch.tensor(self.tensor)
        if self.tensor.ndim != 3:
            # if self.tensor.ndim == 2:
            #     self.tensor = self.tensor.unsqueeze(0)
            # else:
            raise ValueError("The tensor must have exactly three dimensions.")

    @property
    def dim(self) -> int:
        return self.tensor.size(2)

    @property
    def batch_size(self) -> int:
        try:
            return self.tensor.size(0)
        except IndexError as exc:
            raise TypeError(
                "The tensor must have a third dimension (shape=[num_examples x Cardinality x dim])"
            ) from exc

    def __len__(self) -> int:
        """Delegate to the tensor cardinality."""
        return self.tensor.size(1)

    @property
    def shape(self) -> tuple[int]:
        """Delegate to the tensor shape."""
        return self.tensor.shape

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(batch_size={self.batch_size}, "
            f"cardinality={len(self)}, dim={self.dim})"
        )

    @classmethod
    def collate(cls, batch: Sequence[TensorSet]) -> Self:
        return cls(torch.stack([s.tensor for s in batch], dim=0))

    def disperse(self) -> Sequence[TensorSet]:
        return [TensorSet(tensor) for tensor in self.tensor.unbind(0)]

    @classmethod
    def new(cls, tensor: torch.Tensor, **kwargs) -> Self:
        return cls(tensor, **kwargs)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        args = [getattr(a, "tensor", a) for a in args]
        ret = func(*args, **kwargs)
        return ret


@dataclass
class BatchedPaddedTensorSets(BatchedEqualSizedTensorSets):
    """A batch of tensor sets of potentially different cardinalities.

    The tensors representing the sets are padded to the same length. The padding
    is done at the end of the tensor.

    The padding values should be excluded from any computation that is performed
    on the tensor. The padding values should be masked out before any computation
    is performed on the tensor. The start of the padding values is stored in the
    `padding_start_index` attribute.

    Attributes:
        tensor: A tensor of shae [B x N x dim] where B is the number of sets in
          the batch, N is the maximum cardinality of the sets in the batch, and
          dim is the dimensionality of the elements.
        padding_start_index: A list of integers where each integer is the index of
          the first padding value in the corresponding tensor in the batch.
    """

    padding_start_index: list[int]

    @classmethod
    def collate(cls, batch: Sequence[TensorSet]) -> Self:
        tensor = torch.nn.utils.rnn.pad_sequence(
            [s.tensor for s in batch], batch_first=True
        )
        padding_start_index = [len(s) for s in batch]
        return cls(tensor, padding_start_index)

    def disperse(self) -> Sequence[TensorSet]:
        return [
            TensorSet(tensor[:length])
            for tensor, length in zip(self.tensor.unbind(0), self.padding_start_index)
        ]


@dataclass
class BatchedPyGSets(BatchedSets):
    """A batch of PyTorch Geometric sets.

    This is a wrapper around a PyTorch Geometric batch of graphs.

    Attributes:
        batch_data: A PyTorch Geometric batch of graphs.
    """

    batch_data: torch_geometric.data.Batch

    @property
    def dim(self) -> int:
        return self.batch_data.num_node_features

    @property
    def batch_size(self) -> int:
        return self.batch_data.batch_size

    @classmethod
    def collate(cls, batch: list[PyGSet]) -> Self:
        return cls(torch_geometric.data.Batch.from_data_list(batch))

    def disperse(self) -> list[PyGSet]:
        return [PyGSet(s.x) for s in self.batch_data.to_data_list()]


class BatchedStackedTensorSets(BatchedPyGSets):
    """An adaptor of the BatchedPyGSets class for TensorSets.

    Same as BatchedPyGSets but the methods accept and return TensorSets instead of
    PyGSet objects.
    """

    @classmethod
    def collate(cls, batch: list[TensorSet]) -> Self:
        pyg_batch = BatchedPyGSets.collate([s.to_pyg() for s in batch])
        return cls(pyg_batch.batch_data)

    def disperse(self) -> list[TensorSet]:
        return [s.to_tensor() for s in super().disperse()]
