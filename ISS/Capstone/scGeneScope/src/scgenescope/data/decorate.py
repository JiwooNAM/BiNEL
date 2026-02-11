import functools
from typing import Callable


def transformable(cls):
    """A dataset decorator that makes the dataset transformable."""

    class TransformableDataset:

        def __init__(self, *args, transform=None, **kwargs):
            self.transform = transform
            self._dataset = cls(*args, **kwargs)
            # functools.update_wrapper(self, cls)

        def __getitem__(self, index):
            example = self._dataset.__getitem__(index)
            if self.transform is not None:
                example = self.transform(example)
            return example

        def __getattr__(self, name):
            return getattr(self._dataset, name)

    return TransformableDataset


def add_empty_context(func: Callable) -> Callable:
    """Decorator to add an empty context to a reader."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs), {}

    return wrapper
