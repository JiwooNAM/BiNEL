from collections.abc import Sequence
import itertools
from typing import Callable, Iterable
import torch
import torch.nn as nn
from typing import Any

from .base import LitClassifier


class MultiModalMultipleInputClassifier(LitClassifier):
    """A multi-modal classifier.

    This model takes multiple modalities as input and outputs a prediction.

    Attributes:
        encoders: The encoder models.
        classifier: The classifier model.
        loss: The loss function.
    """

    encoders: nn.ModuleList | nn.ModuleDict
    classifier: nn.Module
    loss: nn.Module
    normalize: bool

    def __init__(
        self,
        *encoders: nn.Module,
        classifier: nn.Module,
        add_normalization: bool = False,
        input_dropout: float | list[float | None] | dict[float | None] | None = None,
        hidden_dropout: float | list[float | None] | dict[float | None] | None = None,
        optimizer_factory: Callable[
            [Iterable[torch.nn.Parameter]], torch.optim.Optimizer
        ],
        scheduler_factory: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler],
        custom_normalizers: (
            list[nn.Module | None] | dict[str, nn.Module | None] | None
        ) = None,
        compile: bool,
        strict_inputs: bool = True,
        ignore_names: bool = False,
        **named_encoders: nn.Module,
    ) -> None:
        """Initialize a model instance.

        Args:
            encoders: The encoder models (cannot be mixed with named_encoders).
            classifier: The classifier model.
            normalize: Whether to perform layer normalization on the embeddings
              before classification.
            named_encoders: The encoder models with names (cannot be mixed with
              encoders).
        """
        num_classes = classifier.out_features
        super().__init__(num_classes, optimizer_factory, scheduler_factory, compile)
        # Get encoders
        if encoders:
            if named_encoders:
                raise ValueError("Cannot mix ordered and named encoders.")

            self.encoders = nn.ModuleList(
                encoder or nn.Identity() for encoder in encoders
            )
            if add_normalization:
                if custom_normalizers:
                    raise ValueError(
                        "Cannot provide both add_normalization and normalizer_factories."
                    )
                normalizers = [
                    nn.LayerNorm(encoder.out_features, elementwise_affine=True)
                    for encoder in self.encoders
                ]
            elif custom_normalizers:
                normalizers = [
                    _normalizer or nn.Identity() for _normalizer in custom_normalizers
                ]
            else:
                normalizers = [nn.Identity() for _ in range(len(self.encoders))]
            self.normalizers = nn.ModuleList(normalizers)

            try:
                dropout_p_values = iter(input_dropout)
            except TypeError:
                dropout_p_values = itertools.repeat(input_dropout, len(self.encoders))
            self.dropouts = nn.ModuleList(
                nn.Dropout(p) if p else nn.Identity() for p in dropout_p_values
            )

        elif named_encoders:
            if ignore_names:
                self.encoders = nn.ModuleList(
                    encoder or nn.Identity() for encoder in named_encoders.values()
                )
                if add_normalization:
                    if custom_normalizers:
                        raise ValueError(
                            "Cannot provide both add_normalization and normalizer_factories."
                        )
                    normalizers = [
                        nn.LayerNorm(encoder.out_features, elementwise_affine=True)
                        for encoder in self.encoders
                    ]
                elif custom_normalizers:
                    normalizers = [_normalizer for _normalizer in custom_normalizers]
                else:
                    normalizers = [nn.Identity() for _ in range(len(self.encoders))]
                self.normalizers = nn.ModuleList(normalizers)
                try:
                    # If specified as a dict, use the values
                    dropout_p_values = input_dropout.values()
                except AttributeError:
                    # If specified as a list, use the list
                    dropout_p_values = iter(input_dropout)
                except TypeError:
                    # If specified as a single value, repeat it
                    dropout_p_values = itertools.repeat(
                        input_dropout, len(self.encoders)
                    )
                self.dropouts = nn.ModuleList(
                    nn.Dropout(p) if p else nn.Identity() for p in dropout_p_values
                )
            else:
                self.encoders = nn.ModuleDict(
                    {
                        name: encoder or nn.Identity()
                        for name, encoder in named_encoders.items()
                    }
                )
                if add_normalization:
                    if custom_normalizers:
                        raise ValueError(
                            "Cannot provide both add_normalization and normalizer_factories."
                        )
                    normalizers = {}
                    for name, encoder in self.encoders.items():
                        try:
                            normalizers[name] = nn.LayerNorm(
                                encoder.out_features, elementwise_affine=True
                            )
                        except AttributeError:
                            # See if encoder is a container type like a Sequential
                            try:
                                normalizers[name] = nn.LayerNorm(
                                    encoder[-1].out_features, elementwise_affine=True
                                )
                            except (AttributeError, TypeError) as exc:
                                raise TypeError(
                                    f"{encoder} doesn't support out_features to initialize layernorm."
                                ) from exc

                elif custom_normalizers:
                    normalizers = {
                        name: _normalizer or nn.Identity()
                        for name, _normalizer in custom_normalizers.items()
                    }
                else:
                    normalizers = {name: nn.Identity() for name in self.encoders.keys()}

                self.normalizers = nn.ModuleDict(normalizers)

                try:
                    self.dropouts = nn.ModuleDict(
                        {
                            name: nn.Dropout(p) if p else nn.Identity()
                            for name, p in input_dropout.items()
                        }
                    )
                except AttributeError:
                    self.dropouts = nn.ModuleDict(
                        {
                            name: (
                                nn.Dropout(input_dropout)
                                if input_dropout
                                else nn.Identity()
                            )
                            for name in self.encoders.keys()
                        }
                    )
        else:
            raise ValueError("No encoders provided.")

        self.hidden_dropout = (
            nn.Dropout(hidden_dropout) if hidden_dropout else nn.Identity()
        )
        self.classify = classifier
        self.loss = nn.CrossEntropyLoss(reduction="mean")
        self.strict_inputs = strict_inputs
                # ---- fusion options ----
        self.fusion = "concat"  # or "gated_concat"
        self.proj_dim = 256     # 추천: 128~512
        self.use_gating = False

        # 이름 기반 입력(dict)일 때 모달리티 순서 고정(중요)
        if isinstance(self.encoders, nn.ModuleDict):
            self.modality_names = list(self.encoders.keys())
        else:
            self.modality_names = None

        # LazyLinear: 입력 dim(out_features) 몰라도 첫 forward 때 자동 추론됨
        if isinstance(self.encoders, nn.ModuleDict):
            self.proj = nn.ModuleDict({k: nn.LazyLinear(self.proj_dim) for k in self.modality_names})
        else:
            self.proj = nn.ModuleList([nn.LazyLinear(self.proj_dim) for _ in range(len(self.encoders))])

        # 벡터 게이트 g: (img_proj || rna_proj) -> proj_dim
        self.gate = nn.Sequential(
            nn.Linear(2 * self.proj_dim, self.proj_dim),
            nn.Sigmoid(),
        )


    def forward(
        self, xs: Sequence[torch.Tensor] | dict[str, torch.Tensor] | torch.Tensor
    ) -> torch.Tensor:
        if isinstance(xs, Sequence):
            xs = [
                normalize(encoder(dropout(x)))
                for encoder, x, normalize, dropout in zip(
                    self.encoders, xs, self.normalizers, self.dropouts, strict=True
                )
            ]
        elif isinstance(xs, dict):
            xs = [
                self.normalizers[name](encoder(self.dropouts[name](xs[name])))
                for name, encoder in self.encoders.items()
            ]
        else:
            if self.strict_inputs:
                raise ValueError("Input must be a dict or tuple.")
            else:
                xs = [self.normalizers[0](self.encoders[0](self.dropouts[0](xs)))]

        xs = [self.hidden_dropout(x) for x in xs]
        if isinstance(self.encoders, nn.ModuleDict):
            # dict 입력이면 modality_names 순서대로 xs가 만들어졌다고 가정
            k1, k2 = self.modality_names[0], self.modality_names[1]
            x1 = self.proj[k1](xs[0])   # [B, proj_dim]
            x2 = self.proj[k2](xs[1])   # [B, proj_dim]
        else:
            x1 = self.proj[0](xs[0])
            x2 = self.proj[1](xs[1])

        g = self.gate(torch.cat([x1, x2], dim=1))  # [B, proj_dim] in (0,1)

        # gated concat (추천): 정보 보존이 잘 됨
        x = torch.cat([g * x1, (1.0 - g) * x2], dim=1)  # [B, 2*proj_dim]
        return self.classify(x)

    def unpack_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        # *xs, y = batch
        try:
            y = batch.pop("condition")
            xs = batch
        except (AttributeError, TypeError):
            xs, y = batch
        return xs, y
