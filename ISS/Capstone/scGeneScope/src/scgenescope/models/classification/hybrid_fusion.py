import torch
import torch.nn as nn
from typing import Any, Callable, Iterable

from .base import LitClassifier


class HybridFusionMultipleInputClassifier(LitClassifier):
    """
    Hybrid fusion classifier.

    Expected batch format (dict):
      batch = {
        "imaging": Tensor[B, D_img],
        "rnaseq":  Tensor[B, D_rna],
        "condition": Tensor[B]  (label)
      }

    Forward:
      x_fused  = cat([imaging, rnaseq], dim=1)
      z_img    = enc_imaging(drop(imaging))
      z_rna    = enc_rnaseq(drop(rnaseq))
      z_fused  = enc_fused(drop(x_fused))
      z_all    = cat([z_img, z_fused, z_rna], dim=1)
      logits   = classifier(drop_hidden(z_all))
    """

    def __init__(
        self,
        *,
        rnaseq: nn.Module,
        imaging: nn.Module,
        fused: nn.Module,
        classifier: nn.Module,
        add_normalization: bool = False,
        input_dropout: dict[str, float | None] | None = None,
        hidden_dropout: float | None = None,
        optimizer_factory: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer],
        scheduler_factory: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler],
        compile: bool = False,
        strict_inputs: bool = True,
    ) -> None:
        # classifier must expose out_features like Linear or your MLP wrapper
        num_classes = getattr(classifier, "out_features", None)
        if num_classes is None:
            raise ValueError("classifier must have attribute `out_features`")

        super().__init__(num_classes, optimizer_factory, scheduler_factory, compile)

        self.encoders = nn.ModuleDict(
            {
                "rnaseq": rnaseq or nn.Identity(),
                "imaging": imaging or nn.Identity(),
                "fused": fused or nn.Identity(),
            }
        )

        # Dropout per input stream
        input_dropout = input_dropout or {}
        self.dropouts = nn.ModuleDict(
            {
                "rnaseq": nn.Dropout(input_dropout.get("rnaseq", 0.0))
                if input_dropout.get("rnaseq", 0.0)
                else nn.Identity(),
                "imaging": nn.Dropout(input_dropout.get("imaging", 0.0))
                if input_dropout.get("imaging", 0.0)
                else nn.Identity(),
                "fused": nn.Dropout(input_dropout.get("fused", 0.0))
                if input_dropout.get("fused", 0.0)
                else nn.Identity(),
            }
        )

        self.hidden_dropout = nn.Dropout(hidden_dropout) if hidden_dropout else nn.Identity()

        # Optional LayerNorm on encoder outputs (same spirit as your multimodal.py)
        if add_normalization:
            norms = {}
            for name, enc in self.encoders.items():
                out_dim = getattr(enc, "out_features", None)
                if out_dim is None:
                    # try Sequential-like
                    try:
                        out_dim = enc[-1].out_features
                    except Exception as exc:
                        raise TypeError(
                            f"Encoder `{name}` must expose out_features (or be Sequential-like)."
                        ) from exc
                norms[name] = nn.LayerNorm(out_dim, elementwise_affine=True)
            self.normalizers = nn.ModuleDict(norms)
        else:
            self.normalizers = nn.ModuleDict({k: nn.Identity() for k in self.encoders.keys()})

        self.classify = classifier
        self.loss = nn.CrossEntropyLoss(reduction="mean")
        self.strict_inputs = strict_inputs

    def forward(self, xs: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if not isinstance(xs, dict):
            if self.strict_inputs:
                raise ValueError("Input must be a dict with keys: rnaseq, imaging")
            else:
                xs = {"rnaseq": xs}

        x_img = xs["imaging"]
        x_rna = xs["rnaseq"]
        x_fused = torch.cat([x_img, x_rna], dim=1)

        z_img = self.normalizers["imaging"](self.encoders["imaging"](self.dropouts["imaging"](x_img)))
        z_rna = self.normalizers["rnaseq"](self.encoders["rnaseq"](self.dropouts["rnaseq"](x_rna)))
        z_fused = self.normalizers["fused"](self.encoders["fused"](self.dropouts["fused"](x_fused)))

        z_all = torch.cat([z_img, z_fused, z_rna], dim=1)
        z_all = self.hidden_dropout(z_all)
        return self.classify(z_all)

    def unpack_batch(self, batch: Any) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        # match your existing behavior
        y = batch.pop("condition")
        xs = batch
        return xs, y
