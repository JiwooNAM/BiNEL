from collections.abc import Sequence, Callable
from typing import Self

import torch
import torch.nn as nn

from ..residual import Residual
from .layers import PermutationEquivariantLayer, PermutationInvariantLayer
from .types import TensorSet, BatchedEqualSizedTensorSets


class DeepSet(nn.Module):
    """A deep set network that processes sets of elements.

    This network processes sets of elements through a sequence of layers,
    maintaining permutation invariance or equivariance as specified by
    the layer types.

    Attributes:
        layers: A list of network layers that process the input sets.
    """

    def __init__(
        self,
        dims: Sequence[int | Sequence[int]],
        aggregation: str | Callable[[int], nn.Module] = "sum",
        conditioning: str | Callable[[int], nn.Module] | None = None,
        readout: str | Callable[[int], nn.Module] | None = None,
        residual: bool = False,
    ):
        """Initialize the DeepSet network.

        Args:
            dims: Sequence of dimensions for each layer.
            aggregation: Aggregation strategy for intermediate layers.
            conditioning: Conditioning strategy for intermediate layers.
            readout: Aggregation strategy for the final layer.
            residual: Whether to use residual connections.
        """
        super().__init__()

        self.layers = nn.ModuleList()

        for indx, layer_dims in enumerate(dims):

            # Determine if this is the last layer
            is_last_layer = indx == len(dims) - 1
            # Create appropriate layer type
            if is_last_layer and readout is not None:
                # Use permutation invariant layer with readout for final layer
                layer = PermutationInvariantLayer.with_mlp_blocks(
                    dims=layer_dims,
                    aggregation=readout,
                    normalization="layer",
                )
            else:
                # Use permutation equivariant layer for intermediate layers
                layer = PermutationEquivariantLayer.with_mlp_blocks(
                    dims=layer_dims,
                    aggregation=aggregation,
                    conditioning=conditioning,
                    normalization="layer",
                )

            # Wrap with Residual if needed
            if residual and not is_last_layer:
                next_dims = dims[indx + 1]
                layer = Residual.with_mlp(
                    base=layer,
                    input_dim=layer_dims[0],  # First dimension from layer_dims
                    output_dim=next_dims[0],  # First dimension from next_dims
                    num_layers=1,  # Use a simple 1-layer MLP for projection
                )

            self.layers.append(layer)

    def forward(
        self, elements: TensorSet | BatchedEqualSizedTensorSets
    ) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            elements: Input set of elements.

        Returns:
            Processed tensor.
        """
        x = elements
        for layer in self.layers:
            x = layer(x)
        return x

    @classmethod
    def with_fixed_hidden_dim(
        cls,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        elementwise_depth: int,
        num_layers: int,
        **kwargs,
    ) -> Self:
        """Create a DeepSet with fixed hidden dimensions.

        Args:
            input_dim: Dimension of input features.
            hidden_dim: Dimension of hidden layers.
            output_dim: Dimension of output features.
            elementwise_depth: Number of MLP layers in each elementwise block.
            num_layers: Number of main network layers.
            **kwargs: Additional arguments to pass to DeepSet constructor.

        Returns:
            A DeepSet instance with the specified architecture.

        Raises:
            ValueError: If num_layers is less than 1.
        """
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        # Create dims array with hidden layers of fixed dimension
        dims = []

        if num_layers == 1:
            # Single layer case: direct input to output
            dims.append(
                [input_dim] + [hidden_dim] * (elementwise_depth - 1) + [output_dim]
            )
        else:
            # First layer: input_dim to hidden_dim with elementwise_depth layers
            dims.append([input_dim] + [hidden_dim] * elementwise_depth)

            # Middle layers: hidden_dim to hidden_dim with elementwise_depth layers each
            for _ in range(num_layers - 2):
                dims.append([hidden_dim] * (elementwise_depth + 1))

            # Final layer: hidden_dim to output_dim with elementwise_depth layers
            dims.append([hidden_dim] * elementwise_depth + [output_dim])

        # # Add singleton output dimension as final layer
        # dims.append([output_dim])

        return cls(dims=dims, **kwargs)

    @classmethod
    def with_doubling_hidden_dim(
        cls,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        elementwise_depth: int,
        num_layers: int,
        **kwargs,
    ) -> Self:
        """Create a DeepSet with hidden dimensions that double each layer.

        This factory method creates a DeepSet where the hidden dimension
        doubles at each layer through concatenated conditioning.
        The conditioning strategy is set to "concat" by default.

        Args:
            input_dim: Dimension of input features.
            hidden_dim: Initial hidden dimension before doubling.
            output_dim: Dimension of output features.
            elementwise_depth: Number of MLP layers in each elementwise block.
            num_layers: Number of main network layers.
            **kwargs: Additional arguments to pass to DeepSet constructor.

        Returns:
            A DeepSet instance with the specified architecture.

        Raises:
            ValueError: If num_layers is less than 1.
        """
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        # Create dims array with hidden layers that double in size
        dims = []

        if num_layers == 1:
            # Single layer case: direct input to output
            dims.append(
                [input_dim] + [hidden_dim] * (elementwise_depth - 1) + [output_dim]
            )
        else:
            # First layer: input_dim to hidden_dim with elementwise_depth layers
            dims.append([input_dim] + [hidden_dim] * elementwise_depth)

            # Middle layers: each layer's dimension doubles due to concatenation
            current_dim = hidden_dim
            for _ in range(num_layers - 2):
                current_dim *= 2
                dims.append([current_dim] * (elementwise_depth + 1))

            # Final layer: current_dim to output_dim with elementwise_depth layers
            dims.append([current_dim * 2] * elementwise_depth + [output_dim])

        # # Add singleton output dimension as final layer
        # dims.append([output_dim])

        # Set conditioning to "concat" by default if not specified
        if "conditioning" in kwargs:
            if kwargs["conditioning"] != "concatenated":
                raise ValueError("conditioning must be set to 'concatenated'")
        else:
            kwargs["conditioning"] = "concatenated"

        return cls(dims=dims, **kwargs)
