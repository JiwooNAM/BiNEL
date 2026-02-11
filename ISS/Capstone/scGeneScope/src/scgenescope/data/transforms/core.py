"""Core transforms that extend and build on other transforms."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Callable, Iterator

from .base import Transform, ExampleTransform, Datum
from .mixin import FromItemsMixin
from ...types import Example, Batch


class Map(Transform):
    """Maps a transform to a batch of examples."""

    def __init__(self, transform: ExampleTransform, collate_fn: Callable = list):
        self.transform = transform
        self.collate_fn = collate_fn

    def __call__(self, batch: Iterator[Example]) -> Batch:
        return self.collate_fn(list(map(self.transform, batch)))

    def __repr__(self) -> str:
        _base = super().__repr__()
        # strip the [example-wise] prefix from the self.transform repr
        _instance_repr = repr(self.transform)
        if _instance_repr.startswith("[example-wise]"):
            _instance_repr = _instance_repr[len("[example-wise]") :]
        try:
            args_repr = f"{_instance_repr}, collate_fn={self.collate_fn.__name__}"
        except AttributeError:
            args_repr = f"{_instance_repr}, collate_fn={self.collate_fn}"
        return "[batch-wise]" + _base.format(args_repr)


class Batchify(Transform):

    def __init__(self, example_transform: Transform, collate_fn: Callable = list):
        self._transform = Map(example_transform, collate_fn)

    def __call__(self, batch: Batch) -> Batch:
        return self._transform(batch)

    def __repr__(self) -> str:
        return repr(self._transform)


class Dispatch(dict, Transform):
    """Dispatches a transform to an example based on a key field.

    Attributes:
        self: A map of key to transform.
    """

    def __call__(self, data: Datum) -> Datum:
        """Apply each transform to the field of an example matching its key."""
        result = {}
        try:
            for key, transform in self.items():
                try:
                    # try to get the value from a namedtuple
                    value = getattr(data, key)
                except AttributeError:
                    # try to get the value from a dictionary
                    value = data[key]
                result[key] = transform(value)
        except KeyError as exc:
            raise TypeError(
                f"Invalid {key=} in transforms. All keys need to match the "
                f"fields of an example."
            ) from exc

        try:
            result = data._replace(**result)
        except AttributeError:
            pass

        return result

    def __repr__(self) -> str:
        _base = Transform.__repr__(self)
        transforms_repr = ", ".join(
            f"{key}: {repr(transform)}" for key, transform in self.items()
        )
        return _base.format(transforms_repr)


class Compose(list[Transform], Transform, FromItemsMixin):
    """Creates a transform from a sequence of transforms."""

    def __call__(self, data: Datum) -> Datum:
        for transform in self:
            data = transform(data)
        return data

    def __repr__(self) -> str:
        transforms_repr = " \u2192 ".join(repr(transform) for transform in self)
        return f"[{transforms_repr}]"


class Distribute(list[Transform | None], Transform, FromItemsMixin):
    """Distributes a sequence of transforms to a sequence of examples."""

    def __call__(self, seq_data: list[Datum]) -> list[Datum]:
        return [
            transform(data) if transform else data
            for transform, data in zip(self, seq_data, strict=True)
        ]

    def __repr__(self) -> str:
        transforms_repr = " , ".join(repr(transform) for transform in self)
        return f"[{transforms_repr}]"


class IndexScatter(list[int], Transform):
    """Scatters an input sequence according to integer indices."""

    def __init__(self, *args, squeeze: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.squeeze = squeeze

    def __call__(self, seq_data: Sequence[Datum]) -> Datum | list[Datum]:
        # if there is only one index, return the corresponding datum
        if len(self) == 1 and self.squeeze:
            return seq_data[self[0]]
        # otherwise, return a list of datums
        return [seq_data[i] for i in self]

    def __repr__(self) -> str:
        indices_repr = ", ".join(str(i) for i in self)
        return f"[{indices_repr}]"


class Slice(IndexScatter):
    """Slices an input sequence according to a start and stop index."""

    def __init__(self, start: int, stop: int):
        super().__init__(list(range(start, stop)))


class KeyScatter(list[str], Transform):
    """Scatters an input dictionary or namedtuple according to field keys."""

    def __call__(self, data: dict[str, Example]) -> list[Example]:
        return [data[key] for key in self]

    def __repr__(self) -> str:
        keys_repr = ", ".join(self)
        return f"[{keys_repr}]"


class Chunk(list[int], Transform):
    """Chunks an input sequence according to integer sizes of chunks."""

    def __call__(self, data: Sequence[Example]) -> list[list[Example]]:
        data = list(data)
        return [[data.pop(0) for _ in range(chunk_size)] for chunk_size in self]


class Shatter(list[int], Transform):
    """Shatters an input sequence according to a sequence of integer indices."""

    pass
