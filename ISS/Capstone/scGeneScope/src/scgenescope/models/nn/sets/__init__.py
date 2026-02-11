from enum import StrEnum
from typing import Any, Dict, Type

from .layers import Elementwise, PermutationEquivariantLayer, PermutationInvariantLayer
from .deepset import DeepSet
from .types import (
    TensorSet,
    PyGSet,
    BatchedEqualSizedTensorSets,
    BatchedPaddedTensorSets,
    BatchedPyGSets,
)

__all__ = [
    "Elementwise",
    "PermutationEquivariantLayer",
    "PermutationInvariantLayer",
    "DeepSet",
    "TensorSet",
    "PyGSet",
    "BatchedEqualSizedTensorSets",
    "BatchedPaddedTensorSets",
    "BatchedPyGSets",
]

class SetType(StrEnum):
    """Enumeration of available set types."""
    TENSOR = "tensor"
    PYG = "pyg"
    BATCHED_EQUAL = "batched_equal"
    BATCHED_PADDED = "batched_padded"
    BATCHED_PYG = "batched_pyg"


# Registry mapping set types to their implementations
SET_REGISTRY: Dict[SetType, Type] = {
    SetType.TENSOR: TensorSet,
    SetType.PYG: PyGSet,
    SetType.BATCHED_EQUAL: BatchedEqualSizedTensorSets,
    SetType.BATCHED_PADDED: BatchedPaddedTensorSets,
    SetType.BATCHED_PYG: BatchedPyGSets,
}


def create_set(set_type: SetType, data: Any, **kwargs) -> Any:
    """Create a set instance of the specified type.

    Args:
        set_type: The type of set to create
        data: The data to create the set from
        **kwargs: Additional arguments to pass to the set constructor

    Returns:
        An instance of the specified set type
    """
    set_class = SET_REGISTRY[set_type]

    if set_type == SetType.TENSOR:
        return set_class.new(data, **kwargs)
    elif set_type == SetType.PYG:
        return set_class.from_tensor(data)
    elif set_type == SetType.BATCHED_EQUAL:
        return set_class.new(data, **kwargs)
    elif set_type == SetType.BATCHED_PADDED:
        if isinstance(data, (list, tuple)):
            return set_class.collate(data)
        return set_class.new(data, **kwargs)
    elif set_type == SetType.BATCHED_PYG:
        if isinstance(data, (list, tuple)):
            return set_class.collate(data)
        return set_class.new(data, **kwargs)
    else:
        raise ValueError(f"Unsupported set type: {set_type}")
