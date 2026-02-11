import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class AttentionAggregator(nn.Module):
    """Aggregates embeddings using multi-head attention mechanism.

    This aggregator uses PyTorch's MultiheadAttention layer to compute attention
    between all positions in the sequence, then aggregates using the attention weights.

    Args:
        input_dim: Dimension of input embeddings
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Initialize multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention and aggregate.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Aggregated tensor of shape (batch_size, input_dim)
        """
        # Apply attention
        attn_output, _ = self.attention(x, x, x)

        # Apply layer norm
        attn_output = self.norm(attn_output)

        # Take mean across sequence length
        return attn_output.mean(dim=1)


class TransformerAggregator(nn.Module):
    """Aggregates multiple embeddings using a Transformer encoder architecture.

    This aggregator uses multiple self-attention layers to capture complex
    relationships between different embeddings before averaging them.

    Args:
        input_dim: Dimension of input embeddings
        n_heads: Number of attention heads in transformer
        dim_feedforward: Dimension of feedforward network in transformer
        dropout: Dropout probability
        num_layers: Number of transformer encoder layers
    """

    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
    ):
        super().__init__()

        encoder_layers = TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.network = TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer encoding and mean pooling.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim) or BatchedEqualSizedTensorSets

        Returns:
            Aggregated tensor of shape (batch_size, input_dim)
        """
        # Handle BatchedEqualSizedTensorSets
        if hasattr(x, "tensor"):
            x = x.tensor
        return self.network(x).mean(dim=1)


class GatedAttentionAggregator(nn.Module):
    """Aggregates embeddings using gated attention mechanism.

    Implements a gated attention mechanism where the attention weights are computed
    using both a tanh and sigmoid gate, as described in the paper
    "Gated Attention Networks for Learning on Large and Spatiotemporal Graphs".

    Args:
        input_dim: Dimension of input embeddings
    """

    def __init__(
        self,
        input_dim: int,
    ):
        super().__init__()

        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
        )

        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )

        self.attention_weights = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated attention mechanism.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Aggregated tensor of shape (batch_size, input_dim)
        """
        V = self.attention_V(x)
        U = self.attention_U(x)
        A = self.attention_weights(V * U)
        A = torch.nn.functional.softmax(A.transpose(-1, -2), dim=-1)

        return torch.bmm(A, x).squeeze(dim=1)


class LinearAttentionAggregator(nn.Module):
    """Aggregates embeddings using simple linear attention.

    Applies softmax directly to the input embeddings to compute attention weights,
    then uses these weights to compute a weighted sum of the inputs.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear attention mechanism.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Aggregated tensor of shape (batch_size, input_dim)
        """
        avg_attn = torch.nn.functional.softmax(x, dim=1)
        x_ = x * avg_attn
        return x_.sum(dim=1)


class NonLinearAttentionAggregator(nn.Module):
    """Aggregates embeddings using non-linear attention mechanism.

    Uses a multi-layer perceptron to compute attention weights before
    applying them to aggregate the input embeddings.

    Args:
        input_dim: Dimension of input embeddings
    """

    def __init__(
        self,
        input_dim: int,
    ):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.Tanh(), nn.Linear(input_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply non-linear attention mechanism.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Aggregated tensor of shape (batch_size, input_dim)
        """
        attn = self.attention(x)
        attn = torch.nn.functional.softmax(attn, dim=1)
        x_ = attn * x
        return x_.sum(dim=1)


class MeanPoolingAggregator(nn.Module):
    """Aggregates embeddings by taking their mean.

    Simple aggregation that computes the average of all input embeddings
    along the sequence dimension.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mean of input embeddings.

        Args:
            x: Input tensor of shape (batch_size, num_elements, input_dim)

        Returns:
            Aggregated tensor of shape (batch_size, input_dim)
        """
        return torch.mean(x, dim=1)


class MaxPoolingAggregator(nn.Module):
    """Aggregates embeddings by taking their maximum values.

    Simple aggregation that takes the maximum value of all input embeddings
    along the sequence dimension.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute element-wise maximum of input embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Aggregated tensor of shape (batch_size, input_dim)
        """
        return torch.max(x, dim=1)[0]


class SumPoolingAggregator(nn.Module):
    """Aggregates embeddings by taking their sum.

    Simple aggregation that computes the sum of all input embeddings
    along the sequence dimension.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sum of input embeddings.

        Args:
            x: Input tensor of shape (batch_size, num_elements, input_dim)

        Returns:
            Aggregated tensor of shape (batch_size, input_dim)
        """
        return torch.sum(x, dim=1)
