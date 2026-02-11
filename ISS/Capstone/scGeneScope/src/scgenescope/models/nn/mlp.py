from __future__ import annotations
from typing import Callable, Self, Dict
import warnings

import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP).

    Attributes:
        depth: The depth of the MLP.
        in_features: The number of input features.
        out_features: The number of output features.
        last_linear: The last linear layer.
    """

    def __init__(
        self,
        dims: list[int],
        activation_factory: Callable[[], nn.Module] | None = None,
        normalization: str | None = None,
        activate_last: bool = False,
        normalize_last: bool = False,
        normalize_before_activation: bool = True,
        bias_last: bool = True,
    ) -> None:
        """Initialize a multi-layer perceptron.

        Args:
            dims: A list of dimensions for the layers of the MLP. Each element
              represents the size of a layer, with the first being input size
              and last being output size.
            activation_factory: Callable that returns an activation function
              module to use between layers. If None, no activation is used and
              a warning is issued.
            normalization: Type of normalization to use between layers. Options
              are:
                - "batch": BatchNorm1d
                - "layer": LayerNorm
                - None: No normalization
            activate_last: Whether to apply activation after the final layer.
            normalize_last: Whether to apply normalization after the final layer.
            normalize_before_activation: If True, normalization is applied
              before activation. If False, activation is applied before
              normalization.
            bias_last: Whether to include bias in the final linear layer.

        Raises:
            ValueError: If dims is empty or contains fewer than 2 dimensions.
        """
        super().__init__()
        self._depth = len(dims) - 1
        self.layers = nn.ModuleList()
        self._grad_norms: Dict[str, float] = {}
        self._grad_hooks = []

        if activation_factory is None:
            warnings.warn("No activation function provided for MLP.")

        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # Setup each layer
            if i < self._depth - 1:
                if in_dim is None:
                    self.layers.append(nn.LazyLinear(out_dim, bias=True))
                else:
                    self.layers.append(nn.Linear(in_dim, out_dim, bias=True))

                activate = activation_factory() if activation_factory else None
                normalize = self.get_normalization(out_dim, normalization)

            # Specialize the behavior of last layer
            else:
                self.layers.append(nn.Linear(in_dim, out_dim, bias=bias_last))

                if activation_factory is not None and activate_last:
                    activate = activation_factory()
                else:
                    activate = None

                if normalize_last:
                    normalize = self.get_normalization(out_dim, normalization)
                else:
                    normalize = None

            self.layers.extend(
                filter(
                    None,
                    (
                        [normalize, activate]
                        if normalize_before_activation
                        else [activate, normalize]
                    ),
                )
            )

        # # Register gradient hooks for all linear layers
        # self._register_grad_hooks()

    # def _register_grad_hooks(self) -> None:
    #     """Register hooks to monitor gradient norms."""
    #     for i, layer in enumerate(self.layers):
    #         if isinstance(layer, nn.Linear):
    #             name = f"linear_{i}"

    #             def hook_fn(grad, name=name):
    #                 self._grad_norms[name] = grad.norm().item()
    #                 return grad

    #             hook = layer.weight.register_hook(hook_fn)
    #             self._grad_hooks.append(hook)

    # def get_gradient_norms(self) -> Dict[str, float]:
    #     """Get the current gradient norms for all linear layers.

    #     Returns:
    #         Dictionary mapping layer names to their gradient norms.
    #     """
    #     return self._grad_norms.copy()

    # def clear_gradient_norms(self) -> None:
    #     """Clear the stored gradient norms."""
    #     self._grad_norms.clear()

    # def __del__(self) -> None:
    #     """Clean up gradient hooks when the model is deleted."""
    #     for hook in self._grad_hooks:
    #         hook.remove()

    def get_normalization(self, dim: int, normalization: str) -> nn.Module | None:
        """Get the normalization layer for the MLP."""
        match normalization:
            case "batch":
                normalize = nn.BatchNorm1d(dim, momentum=0.01, eps=0.001)
            case "layer":
                normalize = nn.LayerNorm(dim, elementwise_affine=False)
            case "instance":
                normalize = nn.InstanceNorm1d(dim, eps=1e-05, momentum=0.1)
            case None:
                normalize = None
            case _:
                raise ValueError(f"Invalid normalization: {self.normalization}")
        return normalize

    def __repr__(self):
        dims = []

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                _in = layer.in_features if layer.in_features else "auto"
                _out = layer.out_features
                dims.append(f"{_in} \u2192 {_out}")
            else:
                dims.append(layer.__class__.__name__)

        return f'MLP({", ".join(dims)})'

    @property
    def last_linear(self) -> nn.Linear:
        for layer in reversed(self.layers):
            if isinstance(layer, nn.Linear):
                return layer

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def in_features(self) -> int:
        return self.layers[0].in_features

    @property
    def out_features(self) -> int:
        for layer in reversed(self.layers):
            if isinstance(layer, nn.Linear):
                return layer.out_features

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs

        for layer in self.layers:
            x = layer(x)

        return x

    @classmethod
    def from_fixed_hidden_dim(
        cls,
        input_dim: int | None,
        output_dim: int,
        hidden_dim: int,
        depth: int,
        **kwargs,
    ) -> Self:
        """Creates an MLP with the same hidden dimension for all hidden layers.

        Args:
            input_dim: Number of input features. If None, will be inferred at runtime.
            output_dim: Number of output features
            hidden_dim: Size of each hidden layer
            depth: Number of layers including input and output
            **kwargs: Additional arguments passed to MLP constructor including:
              - activation: Activation function between layers
              - activate_last: Whether to activate the output layer
              - bias_last: Whether to add bias to output layer

        Returns:
            A new MLP instance with the specified architecture
        """
        dims = [input_dim] + [hidden_dim] * (depth - 1) + [output_dim]
        return cls(dims=dims, **kwargs)
