# Copyright (c) 2024 Altos Labs, Inc. Altos Confidential.

from .core import Transform


class NotNone(Transform):
    """Check if the input is not None.

    Raises:
        ValueError: If the input is None.
    """

    def __call__(self, value):
        if value is None:
            raise ValueError("Detected a None values.")
        return value

    def __repr__(self):
        return "NotNone"
