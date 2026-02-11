"""Residual connection wrapper for neural network modules."""

from typing import Protocol, Self

import torch
import torch.nn as nn

from .mlp import MLP


class ProjectionFactory(Protocol):
    """Protocol defining the interface for projection factories.

    A projection factory must be callable with input_dim and output_dim as
    keyword arguments, plus optional additional kwargs, and return an nn.Module.
    """

    def __call__(self, *, input_dim: int, output_dim: int, **kwargs) -> nn.Module: ...


class Residual(nn.Module):
    """A wrapper module that applies residual connections to its wrapped module.

    This module takes a base module and applies residual connections in its forward pass.
    If the input and output dimensions match, it adds them directly. If they don't match,
    it applies a projection layer to match the dimensions before adding.

    Attributes:
        base: The base module to wrap with residual connections.
        projection: Optional projection layer to match dimensions if needed.
    """

    base: nn.Module
    projection: nn.Module | None

    def __init__(
        self,
        base: nn.Module,
        projection: nn.Module | None = None,
    ):
        """Initialize the Residual wrapper.

        Args:
            base: The base module to wrap with residual connections.
            projection: Optional projection layer to match dimensions if needed.
        """
        super().__init__()
        self.base = base
        self.projection = projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor.

        Returns:
            Output tensor with residual connection applied.
        """
        # Store input for residual connection
        residual = x

        # Apply base module
        out = self.base(x)

        # Apply projection if needed
        if self.projection is not None:
            residual = self.projection(residual)

        # Add residual connection
        return out + residual

    @classmethod
    def from_projection_factory(
        cls,
        base: nn.Module,
        projection_factory: ProjectionFactory,
        input_dim: int,
        output_dim: int,
        **projection_kwargs,
    ) -> Self:
        """Create a Residual module using a projection factory.

        This is a factory method that creates a Residual module with a projection layer
        created using the provided factory. The projection layer is only created if
        input and output dimensions don't match.

        Args:
            base: The base module to wrap with residual connections.
            projection_factory: Class to use for creating the projection layer.
            input_dim: Dimension of the input tensor.
            output_dim: Dimension of the output tensor.
            **projection_kwargs: Additional arguments to pass to the projection factory.

        Returns:
            A Residual module with projection if needed.
        """
        # Create projection layer if dimensions don't match
        if input_dim != output_dim:
            projection = projection_factory(
                input_dim=input_dim, output_dim=output_dim, **projection_kwargs
            )
        else:
            projection = None

        return cls(base=base, projection=projection)

    @classmethod
    def with_mlp(
        cls,
        base: nn.Module,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        hidden_dim: int | None = None,
        **projection_kwargs,
    ) -> Self:
        """Create a Residual module with MLP projection.

        This is a convenience method for creating a Residual module that uses
        MLP layers for dimension matching. The MLP can learn more complex
        transformations than simple linear projections.

        Args:
            base: The base module to wrap with residual connections.
            input_dim: Dimension of the input tensor.
            output_dim: Dimension of the output tensor.
            hidden_dim: Optional hidden dimension for the MLP. If None, uses
                the average of input and output dimensions.
            num_layers: Number of layers in the MLP projection.
            **projection_kwargs: Additional arguments to pass to MLP.

        Returns:
            A Residual module with MLP projection if needed.
        """

        # Use average of input/output dims as hidden dim if not specified
        if hidden_dim is None and num_layers > 1:
            hidden_dim = (input_dim + output_dim) // 2

        return cls.from_projection_factory(
            base=base,
            projection_factory=MLP.from_fixed_hidden_dim,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            depth=num_layers,
            **projection_kwargs,
        )
