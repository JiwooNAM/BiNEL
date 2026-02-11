from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NamedTuple, Sequence, Protocol

from omegaconf import DictConfig


Indexed = Sequence
Spec = dict[str, Any] | DictConfig


class DataSplitIndexedTuple(NamedTuple):
    """A named tuple for the split data."""

    train: Any
    val: Any
    test: Any


@dataclass(slots=True)
class DataSplitIndexedDict:
    """A dataclass for the split data."""

    train: Any
    val: Any
    test: Any


class Index(StrEnum):
    """Enum class for indexing modes."""

    LOC = "loc"
    CONDITION = "condition"


class SupportsGrouping(Protocol):
    """A protocol for objects that can be grouped by observations."""

    def groupby(self, axis: str) -> Any: ...


class SupportsIndexModeSelect(Protocol):
    """A protocol for objects that support indexing mode selection."""

    indexing: Index
    conditions: Sequence[str]

    def __getitems__(self, index: int | str) -> Any: ...


class Example:
    pass


class Batch:
    pass


class ExampleMultiLabel:
    pass


class BatchMultiLabel:
    pass
