from typing import Any, Callable

import torch.nn as nn

from .conditioning import ContextualMLP, Concatenated, IgnoreContext
from .aggregators import (
    TransformerAggregator,
    GatedAttentionAggregator,
    LinearAttentionAggregator,
    NonLinearAttentionAggregator,
    MeanPoolingAggregator,
    MaxPoolingAggregator,
    SumPoolingAggregator,
)


class AggregationMixin:
    """Mixin that provides aggregation functionality to a class.

    This mixin provides a class method to create aggregation modules from strings
    or callables, making it easy to add consistent aggregation behavior to any
    class that needs it.

    An aggregator is a callable that takes a tensor of shape
    (batch_size, num_elements, dim) and returns a tensor of shape
    (batch_size, dim).

    Note: num_elements is the number of elements in the set also known as the
    cardinality.

    Available built-in aggregators:
        - "sum": Sum over cardinality dimension
        - "mean": Mean over cardinality dimension
        - "max": Max over cardinality dimension
        - "transformer": Transformer-based aggregation
        - "gated_attention": Gated attention mechanism
        - "linear_attention": Linear attention mechanism
        - "nonlinear_attention": Non-linear attention mechanism
    """

    @classmethod
    def build_aggregator(
        cls,
        aggregator: str | Callable[[int], nn.Module],
        dim: int | None = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Create an aggregation module from a string or callable.

        Args:
            aggregator: Either a string name of the aggregator or a callable
                that optionally takes a dimension and returns an aggregation
                module.
            dim: Input dimension for the aggregator. Required for some aggregator
                types.
            **kwargs: Additional keyword arguments to pass to the aggregator
                constructor.

        Returns:
            An aggregation module.

        Raises:
            ValueError: If aggregator type is unknown or if dim is required but
                not provided.
        """
        match aggregator:
            case str():
                match aggregator.lower():
                    case "sum":
                        return SumPoolingAggregator()
                    case "mean":
                        return MeanPoolingAggregator()
                    case "max":
                        return MaxPoolingAggregator()
                    case "linear_attention":
                        return LinearAttentionAggregator()
                    case "transformer":
                        if dim is None:
                            raise ValueError(
                                "dim must be specified for transformer aggregator"
                            )
                        return TransformerAggregator(
                            input_dim=dim,
                            n_heads=kwargs.get("n_heads", 4),
                            dim_feedforward=kwargs.get("dim_feedforward", dim * 4),
                            dropout=kwargs.get("dropout", 0.1),
                            num_layers=kwargs.get("num_layers", 2),
                        )
                    case "gated_attention":
                        if dim is None:
                            raise ValueError(
                                "dim must be specified for gated_attention aggregator"
                            )
                        return GatedAttentionAggregator(input_dim=dim)
                    case "nonlinear_attention":
                        if dim is None:
                            raise ValueError(
                                "dim must be specified for nonlinear_attention aggregator"
                            )
                        return NonLinearAttentionAggregator(input_dim=dim)
                    case _:
                        raise ValueError(f"Unknown aggregator type: {aggregator}")
            case _:
                try:
                    return aggregator(dim, **kwargs)
                except TypeError:
                    # If aggregator doesn't accept dim parameter, call without it
                    return aggregator(**kwargs)


class ConditioningMixin:
    """Mixin that provides conditioning functionality to a class.

    This mixin provides a class method to create conditioning modules from strings
    or callables, making it easy to add consistent conditioning behavior to any
    class that needs it.
    """

    @classmethod
    def build_conditioning(
        cls,
        conditioning: str | Callable[[int], nn.Module] | None,
        dim: int | None = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Create a conditioning module from a string or callable.

        Args:
            conditioning: Either a string name of the conditioning module, a callable that
                optionally takes a dimension and returns a conditioning module, or None
            dim: Input dimension for the conditioning module. Required only if conditioning
                is a callable that needs the dimension
            **kwargs: Additional keyword arguments to pass to the conditioning module

        Returns:
            A conditioning module
        """
        match conditioning:
            case None:
                return IgnoreContext()
            case str():
                match conditioning.lower():
                    case "concatenated":
                        return Concatenated()
                    case "contextualmlp":
                        return ContextualMLP.from_fixed_hidden_dim(
                            input_dim=dim * 2,
                            output_dim=dim,
                            hidden_dim=dim,
                            depth=2,
                            **kwargs,
                        )
                    case _:
                        raise ValueError(f"Unknown conditioning type: {conditioning}")
            case _:
                try:
                    return conditioning(dims=dim, **kwargs)
                except TypeError:
                    # If conditioning doesn't accept dim parameter, call without it
                    return conditioning(**kwargs)


class ActivationMixin:
    """Mixin that provides activation functionality to a class.

    This mixin provides a class method to create activation modules from strings
    or callables, making it easy to add consistent activation behavior to any
    class that needs it.
    """

    @classmethod
    def build_activation(
        cls,
        activation: str | Callable[[], nn.Module] | None,
        **kwargs: Any,
    ) -> nn.Module:
        """Create an activation module from a string or callable.

        Args:
            activation: Either a string name of the activation function, a callable that
                returns an activation module, or None
            **kwargs: Additional keyword arguments to pass to the activation module

        Returns:
            An activation module
        """
        match activation:
            case None:
                return nn.ReLU()
            case str():
                match activation.lower():
                    case "relu":
                        return nn.ReLU(**kwargs)
                    case "leakyrelu":
                        return nn.LeakyReLU(**kwargs)
                    case "tanh":
                        return nn.Tanh(**kwargs)
                    case "sigmoid":
                        return nn.Sigmoid(**kwargs)
                    case "gelu":
                        return nn.GELU(**kwargs)
                    case _:
                        raise ValueError(f"Unknown activation type: {activation}")
            case _:
                try:
                    return activation(**kwargs)
                except TypeError:
                    # If activation doesn't accept kwargs, call without them
                    return activation()


class NormalizationMixin:
    """Mixin that provides normalization functionality to a class.

    This mixin provides a class method to create normalization modules from strings
    or callables, making it easy to add consistent normalization behavior to any
    class that needs it.
    """

    @classmethod
    def build_normalization(
        cls,
        normalization: str | Callable[[], nn.Module] | None,
        dim: int | None = None,
        **kwargs: Any,
    ) -> nn.Module | None:
        """Create a normalization module from a string or callable.

        Args:
            normalization: Either a string name of the normalization function, a callable that
                returns a normalization module, or None
            dim: The dimension to normalize over. Required for some normalization types.
            **kwargs: Additional keyword arguments to pass to the normalization module

        Returns:
            A normalization module, or None if normalization is None
        """
        match normalization:
            case None:
                return None
            case str():
                if dim is None:
                    raise ValueError(
                        "dim must be specified for built-in normalization types"
                    )

                match normalization.lower():
                    case "batch":
                        return nn.BatchNorm1d(dim, **kwargs)
                    case "layer":
                        return nn.LayerNorm(dim, **kwargs)
                    case "instance":
                        return nn.InstanceNorm1d(dim, **kwargs)
                    case _:
                        raise ValueError(f"Unknown normalization type: {normalization}")
            case _:
                try:
                    return normalization(**kwargs)
                except TypeError:
                    # If normalization doesn't accept kwargs, call without them
                    return normalization()
