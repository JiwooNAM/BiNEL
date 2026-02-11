from collections.abc import Callable, Sequence
import torch
import torch.nn as nn


class Miso(torch.nn.Module):

    encoders: nn.ModuleList | nn.ModuleDict
    classifier: nn.Module
    loss: nn.Module
    normalize: bool

    def __init__(
        self,
        *encoders: nn.Module | None,
        readout: nn.Module | Callable,
        add_normalization: bool = False,
        input_dropout: float | None = None,
        custom_normalizers: (
            list[nn.Module | None] | dict[str, nn.Module | None] | None
        ) = None,
        ignore_names: bool = False,
        out_features: int | None = None,
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
        super().__init__()

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
                    normalizers = {
                        name: nn.LayerNorm(
                            encoder.out_features, elementwise_affine=True
                        )
                        or nn.Identity()
                        for name, encoder in self.encoders.items()
                    }
                elif custom_normalizers:
                    normalizers = {
                        name: _normalizer or nn.Identity()
                        for name, _normalizer in custom_normalizers.items()
                    }
                else:
                    normalizers = {name: nn.Identity() for name in self.encoders.keys()}

                self.normalizers = nn.ModuleDict(normalizers)
        else:
            raise ValueError("No encoders provided.")

        self.dropout = nn.Dropout(input_dropout) if input_dropout else nn.Identity()
        self.readout = readout
        self.out_features = out_features

    def forward(
        self, xs: Sequence[torch.Tensor] | dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(xs, Sequence):
            xs = [
                normalize(encoder(self.dropout(x)))
                for encoder, x, normalize in zip(
                    self.encoders, xs, self.normalizers, strict=True
                )
            ]
        elif isinstance(xs, dict):
            xs = [
                self.normalizers[name](encoder(self.dropout(xs[name])))
                for name, encoder in self.encoders.items()
            ]
        else:
            raise ValueError("Input must be a dict or tuple.")
        return self.readout(xs)
