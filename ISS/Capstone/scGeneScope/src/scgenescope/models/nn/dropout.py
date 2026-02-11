import torch.nn as nn


class DropoutModule(nn.Module):
    """A simple wrapper around torch.nn.Dropout.

    This module applies dropout to the input tensor during training, randomly zeroing
    some of the elements with probability p.

    Args:
        dropout_rate (float, optional): Probability of an element to be zeroed.
            Default: 0.5
    """

    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        return self.dropout(x)
