# Copyright (c) 2024 Altos Labs, Inc. Altos Confidential.

from typing import Any, Callable

import numpy as np
import torch
from scipy.sparse import csr_matrix

from .core import Transform


class ToDense(Transform):
    """Convert a sparse matrix/tensor to a dense matrix/tensor."""

    def __call__(self, value: torch.Tensor | csr_matrix) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            output = value.to_dense()
        elif isinstance(value, csr_matrix):
            output = torch.Tensor(value.toarray())
        else:
            raise TypeError(
                f"Invalid type for {value=}. Must be either a tensor or a csr_matrix."
            )
        return output

    def __repr__(self):
        return "ToDense"


class ToFloat(Transform):
    """Convert a tensor to float."""

    def __call__(self, value: torch.Tensor):
        return value.float()

    def __repr__(self):
        return "ToFloat"


class Squeeze(Transform):
    """Squeeze a tensor."""

    def __call__(self, value: torch.Tensor):
        return value.squeeze()

    def __repr__(self):
        return super().__repr__()


class ToTensor(Transform):
    """Convert an input to a tensor."""

    def __init__(self, squeeze: bool = True, dtype: str | None = None) -> None:
        super().__init__()
        self.squeeze = squeeze
        self.dtype = getattr(torch, dtype) if dtype else None

    def __call__(self, value: Any) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            value = torch.from_numpy(np.asarray(value))
        tensor = value.to(dtype=self.dtype)
        if self.squeeze:
            tensor = tensor.squeeze()
        return tensor

    def __repr__(self) -> str:
        squeezed_status = ", squeezed" if self.squeeze else ""
        return f"ToTensor(dtype={self.dtype}{squeezed_status})"


class MapApply(Transform):
    """Map each transform to an input based on a key.

    Attributes:
        transform_map: A map of key to transform.
    """

    transform_map: dict[str, Transform | Callable]

    def __init__(
        self,
        transforms: dict[str, Transform | Callable],
        init_params_map: dict | None = None,
    ) -> None:
        """Initializes the instance based on passed transforms.

        This classes supports two ways of initializing the transforms. The first
        is by passing a map of key to transform. The second is by passing a map of
        key to factory callable. The factory callable will be called with the
        corresponding init params from the init_params_map. The factory callable
        should return a Transform.

        Args:
            transforms: A map of key to transform.
            init_params_map: A map of key to init params for the transforms.

        Raises:
            ValueError: If init_params_map is not None when using a dict of
                Transforms.
            TypeError: If the transform is not a dict of Transform or a callable.
        """
        super().__init__()
        self.transform_map = {}
        for key, transform in transforms.items():
            # Transforms are dict[str, Transform], directly assign them
            if isinstance(transform, Transform):
                if init_params_map is not None:
                    raise ValueError(
                        "init_params_map should be None when using a dict of "
                        "Transforms."
                    )
                self.transform_map[key] = transform
            # Transforms are dict[str, factory_callable], call the factory
            elif callable(transform):
                self.transform_map[key] = transform(init_params_map[key])
            else:
                raise TypeError(
                    f"Invalid type for {key=} in transform. Must be either a "
                    f"Transform or a callable."
                )

    def __call__(self, value_map: dict[str, Any]) -> dict[str, Any]:
        return {key: self.transform_map[key](val) for key, val in value_map.items()}

    def __repr__(self) -> str:
        transforms_repr = ", ".join(
            f"{key}: {repr(transform)}" for key, transform in self.transform_map.items()
        )
        return "{" + transforms_repr + "}"


class NoOp(Transform):
    """No operation transform."""

    def __call__(self, value: Any) -> Any:
        return value

    def __repr__(self) -> str:
        return "NoOp"
