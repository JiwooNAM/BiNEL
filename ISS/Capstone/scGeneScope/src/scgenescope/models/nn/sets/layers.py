from collections.abc import Sequence
from typing import Self, Callable, Any

import torch
import torch.nn as nn

from ..mlp import MLP
from ..residual import Residual
from ..mixin import AggregationMixin, ConditioningMixin


class Elementwise(MLP):
    """An elementwise set mlp function.

    This class extends MLP to provide a more intuitive interface for processing
    set elements. It automatically sets ReLU as the default activation function.
    """

    def __init__(self, dims: Sequence[int], **kwargs):
        """Initialize the elementwise layer.

        Args:
            dims: Sequence of dimensions for the MLP layers.
            **kwargs: Additional arguments to pass to MLP constructor.
        """
        if "activation_factory" not in kwargs:
            kwargs["activation_factory"] = nn.ReLU
        super().__init__(dims=dims, **kwargs)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply MLP to input features.

        Args:
            features: Input tensor of shape (batch_size, num_elements, dim).

        Returns:
            Processed tensor of shape (batch_size, num_elements, dim).
        """
        return super().forward(inputs=features)


class _Contextualize(nn.Module):
    """A wrapper module that contextualizes a set of features."""

    def __init__(self, aggregator: nn.Module, conditioner: nn.Module):
        super().__init__()
        self.aggregator = aggregator
        self.conditioner = conditioner

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        context = self.aggregator(features)  # (batch_size, dim)
        return self.conditioner(
            features, context
        )  # (batch_size, num_elements, new_dim)


class PermutationInvariantLayer(nn.Module, AggregationMixin):
    """A layer that maintains permutation invariance through aggregation.

    Attributes:
        elementwise: The network that processes each element independently.
        aggregator: The aggregation module used by the layer.
    """

    def __init__(
        self,
        elementwise: nn.Module,
        aggregator: nn.Module,
    ):
        """Initialize the permutation invariant layer.

        Args:
            elementwise: Module to use for elementwise processing.
            aggregator: Module to use for aggregation.
        """
        super().__init__()
        self.elementwise = elementwise
        self.aggregator = aggregator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply elementwise transformation and aggregation.

        Args:
            x: Input tensor of shape (batch_size, num_elements, dim).

        Returns:
            Aggregated tensor of shape (batch_size, dim).
        """
        x = self.elementwise(x)
        return self.aggregator(x)

    @classmethod
    def with_mlp_blocks(
        cls,
        dims: Sequence[int],
        aggregation: str | Callable[[int], nn.Module] = "mean",
        **elementwise_kwargs: Any,
    ) -> Self:
        """Create a permutation invariant layer using MLP-based blocks.

        This factory method handles the construction of all components with
        proper dimension handling to ensure consistency.

        Args:
            dims: Sequence of dimensions for the MLPs.
            aggregation: Aggregation strategy to use.
            **elementwise_kwargs: Additional keyword arguments to pass to
                elementwise MLP.

        Returns:
            A permutation invariant layer with MLP-based elementwise network.
        """
        elementwise = Elementwise(dims=dims, **(elementwise_kwargs or {}))
        aggregator = cls.build_aggregator(aggregation, dim=dims[-1])
        return cls(elementwise=elementwise, aggregator=aggregator)


class PermutationEquivariantLayer(nn.Module, AggregationMixin, ConditioningMixin):
    """A layer that maintains permutation equivariance through conditioning.

    Attributes:
        elementwise: The network that processes each element independently.
        aggregator: The aggregation module used by the layer.
        conditioner: The conditioning module used by the layer.
    """

    def __init__(
        self,
        elementwise: nn.Module,
        aggregator: nn.Module,
        conditioner: nn.Module,
        residual: bool = False,
        **residual_kwargs: Any,
    ):
        """Initialize the permutation equivariant layer.

        Args:
            elementwise: Module to use for elementwise processing.
            aggregator: Module to use for aggregation.
            conditioner: Module to use for conditioning.
            residual: Whether to use a residual connection.
            **residual_kwargs: Additional keyword arguments to pass to
                residual connection.
        """
        super().__init__()
        self.elementwise = elementwise
        self.contextualize = _Contextualize(
            aggregator=aggregator, conditioner=conditioner
        )
        if residual:
            if residual_kwargs:
                self.contextualize = Residual.with_mlp(
                    base=self.contextualize,
                    **residual_kwargs,
                )
            else:
                self.contextualize = Residual(self.contextualize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply elementwise transformation, aggregation, and conditioning.

        Args:
            x: Input tensor of shape (batch_size, num_elements, dim).

        Returns:
            Conditioned tensor of shape (batch_size, num_elements, dim).
        """
        x = self.elementwise(x)  # (batch_size, num_elements, new_dim)
        return self.contextualize(x)  # (batch_size, num_elements, new_dim)

    @classmethod
    def with_mlp_blocks(
        cls,
        dims: Sequence[int],
        aggregation: str | Callable[[int], nn.Module] = "mean",
        conditioning: str | Callable[[int], nn.Module] | None = None,
        residual: bool = False,
        **elementwise_kwargs: Any,
    ) -> Self:
        """Create a permutation equivariant layer using MLP-based blocks.

        This factory method handles the construction of all components with
        proper dimension handling to ensure consistency.

        Args:
            dims: Sequence of dimensions for the MLPs.
            aggregation: Aggregation strategy to use.
            conditioning: Conditioning strategy to use.
            **elementwise_kwargs: Additional keyword arguments to pass to
                elementwise MLP.

        Returns:
            A permutation equivariant layer with MLP-based elementwise and
            conditioning networks.
        """
        elementwise = Elementwise(dims=dims, **(elementwise_kwargs or {}))
        aggregator = cls.build_aggregator(aggregation, dim=dims[-1])
        conditioner = cls.build_conditioning(conditioning, dim=dims[-1])
        return cls(
            elementwise=elementwise,
            aggregator=aggregator,
            conditioner=conditioner,
            residual=residual,
        )
