import torch

from .mlp import MLP


class Concatenated(torch.nn.Module):
    """Concatenate the input with the context."""

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Concatenate context to the last dimension of x.

        The context is broadcasted to match the dimensions of x.

        Args:
            x: The input tensor typically of shape (batch_size, num_elements, input_dim).
            context: The context tensor typically of shape (batch_size, context_dim)
                or (batch_size, num_elements, context_dim) if the context is not shared.

        Returns:
            The concatenated tensor of shape (batch_size, num_elements, input_dim + context_dim).
        """
        # Handle both shared and non-shared contexts
        if context.dim() == 2:  # Shared context: (batch_size, context_dim)
            # Add dimension for elements and expand
            context = context.unsqueeze(1).expand(-1, x.size(1), -1)
        elif (
            context.dim() == 3
        ):  # Non-shared context: (batch_size, num_elements, context_dim)
            # Already in correct shape, no need to expand
            pass
        else:
            raise RuntimeError(
                f"Context tensor must have 2 or 3 dimensions, got {context.dim()}"
            )

        return torch.cat([x, context], dim=-1)


class ContextualMLP(Concatenated, MLP):
    """An MLP that processes input conditioned on context."""

    def __init__(self, **kwargs):
        if "activation_factory" not in kwargs:
            kwargs["activation_factory"] = torch.nn.ReLU
        super().__init__(**kwargs)

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return MLP.forward(self, Concatenated.forward(self, x, context))


class IgnoreContext(torch.nn.Module):
    """Ignore the context."""

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return x
