from collections.abc import Callable
from typing import Dict, Optional, Tuple, Type, Union

from torch.utils.data._utils.collate import collate, default_collate_fn_map
from torch.utils.data import default_collate
from torch import Tensor


def collate_series_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):

    pass


def noop_collate(batch):
    return batch


def extended_collate(batch):
    collate_map = default_collate_fn_map.copy()
    try:
        import dgl

        collate_map.update(
            {
                dgl.DGLGraph: collate_graph_fn,
            }
        )
    except ImportError:
        pass

    return collate(batch, collate_fn_map=collate_map)
