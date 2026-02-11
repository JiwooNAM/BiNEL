import torch

from .collate import extended_collate, noop_collate


class ExtendedDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, collate_fn=extended_collate, **kwargs)


class NoCollateDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, collate_fn=noop_collate, **kwargs)
