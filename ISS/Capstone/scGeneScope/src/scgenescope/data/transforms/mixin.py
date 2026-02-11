from collections.abc import Iterable
import itertools
from typing import Self

from .core import Transform
from .ops import NoOp


class FromItemsMixin:

    @classmethod
    def from_items(
        cls,
        *items: Transform | None,
        transforms: Iterable[Transform | None] | None = None,
        **named_items: Transform | None,
    ) -> Self:
        return cls(itertools.chain(items, transforms or (), named_items.values()))

    @classmethod
    def repeat(cls, transform: Transform, times: int) -> Self:
        if times:
            return cls(itertools.repeat(transform, times))
        else:
            # return NoOp()
            return transform
