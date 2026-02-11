from typing import Any, Callable, Iterable

import torch

from .base import LitClassifier


class UnimodalSampleClassifier(LitClassifier):
    """A unimodal classifier.

    This model takes a single modality as input and outputs a prediction.

    Attributes:
        encoder: The encoder model.
        classifier: The classifier model.
    """

    def __init__(
        self,
        encoder: torch.nn.Module | None,
        classifier: torch.nn.Module,
        optimizer_factory: Callable[
            [Iterable[torch.nn.Parameter]], torch.optim.Optimizer
        ],
        scheduler_factory: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler],
        compile: bool,
    ) -> None:
        num_classes = classifier.out_features
        super().__init__(num_classes, optimizer_factory, scheduler_factory, compile)
        self.embed = encoder or torch.nn.Identity()
        self.classify = classifier

    def forward(self, x):
        x = self.embed(x)
        return self.classify(x)

    def unpack_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        return x, y
