from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# import dgl
from torch import Tensor

from .base import Transform
from ...models.nn.sets import create_set, SetType

__all__ = ["ToSet"]


# @dataclass
# class ToEdgelessGraph(Transform):
#     """Embed the features into a graph without edges."""

#     keys: list[str] | str = "features"

#     def __post_init__(self):
#         # Strip the list if there is only one key
#         if not isinstance(self.keys, str) and len(self.keys) == 1:
#             self.keys = self.keys[0]

#     def __call__(self, features: Tensor | Sequence[Tensor]) -> dgl.DGLGraph:
#         try:
#             num_nodes = features.shape[0]
#             ndata = {self.keys: features}
#         except AttributeError:
#             num_nodes = features[0].shape[0]
#             ndata = {key: features[i] for i, key in enumerate(self.keys)}

#         graph = dgl.graph(([], []), num_nodes=num_nodes)
#         graph.ndata.update(ndata)
#         return graph


@dataclass
class ToSet(Transform):
    """Convert a tensor into a set representation.

    Args:
        set_type: The type of set to convert to as SetType enum
    """

    set_type: SetType

    def __post_init__(self):
        self.set_type = SetType(self.set_type)

    def __call__(self, tensor: Tensor) -> Any:
        return create_set(self.set_type, tensor)

    def __repr__(self):
        return f"ToSet(set_type={self.set_type})"
