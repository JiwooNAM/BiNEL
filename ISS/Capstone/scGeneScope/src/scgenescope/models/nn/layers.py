from collections.abc import Callable
from typing import Any
import torch


class Operator(torch.nn.Module):

    def __init__(self, operation: Callable) -> None:
        super().__init__()
        self.op = operation

    def forward(self, inputs: Any) -> Any:
        return self.op(inputs)

    def __repr__(self) -> str:
        return f"Operator({self.op})"
