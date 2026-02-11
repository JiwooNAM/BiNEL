from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TypeVar

from ...types import Example, Batch


Datum = TypeVar("Datum", Example, Batch)


class Transform(ABC):
    """Abstract transform interface."""

    @abstractmethod
    def __call__(self, data: Datum) -> Datum:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        return self.__class__.__name__ + "({!s})"


class ExampleTransform(Transform):
    """Transforms an example."""

    @abstractmethod
    def __call__(self, example: Example) -> Example:
        pass

    def __repr__(self) -> str:
        _base = super().__repr__()
        return "[example-wise]" + _base

    # def batchify(self, collate_fn: callable = list) -> Map:
    #     """Converts an example transform to a batch transform."""
    #     return Map(self, collate_fn)
