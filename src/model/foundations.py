"""Foundational Neural Network primitives built from scratch in PyTorch.

Implements:
1. CustomLinear: Affine transformation with GPT-style initialization
2. CustomEmbedding: Discrete table lookup with tensor indexing
3. CustomGELU: Gaussian Error Linear Unit (exact & tanh approximation)
4. CustomReLU: Rectified Linear Unit
5. CustomSoftmax: Numerically stable exponential normalization
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Optional


class CustomLinear(nn.Module):
    """Linear (Affine) transformation layer: y = x @ W.T + b

    Mathematically:
        Given x in R^(... x d_in), W in R^(d_out x d_in), and b in R^(d_out):
        y_j = sum_{k=1}^{d_in} x_k * W_{j, k} + b_j
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        std: float = 0.02,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight matrix shaped (out_features, in_features) following PyTorch conventions
        # GPT architecture typically initializes linear projection weights with Normal(mean=0.0, std=0.02)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias: Optional[nn.Parameter] = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self._init_weights(std)

    def _init_weights(self, std: float) -> None:
        """Initializes weights using truncated/normal distribution."""
        nn.init.normal_(self.weight, mean=0.0, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (..., in_features)

        Returns:
            Tensor of shape (..., out_features)
        """
        # Linear projection: x @ W.T
        out = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            out = out + self.bias
        return out


class CustomEmbedding(nn.Module):
    """Embedding layer mapping discrete integer IDs to dense continuous vectors.

    Mathematically:
        W in R^(V x d_model)
        For input token index t in {0, ..., V-1}, output vector e_t = W[t]
        Equivalent to: one_hot(t) @ W
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        std: float = 0.02,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Weight table of shape (num_embeddings, embedding_dim)
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight, mean=0.0, std=std)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Args:
            indices: LongTensor of arbitrary shape (*shape) with values in [0, num_embeddings - 1]

        Returns:
            Tensor of shape (*shape, embedding_dim)
        """
        if not indices.dtype in (torch.int64, torch.int32):
            raise TypeError(f"Embedding indices must be integer types, got {indices.dtype}")
        # Efficient vector gather lookup: self.weight[indices]
        return self.weight[indices]


class CustomGELU(nn.Module):
    """Gaussian Error Linear Unit (GELU) activation function.

    Used universally in GPT-2, GPT-3, BERT, and modern LLMs.

    Exact formula:
        GELU(x) = x * Phi(x) = x * P(X <= x) where X ~ N(0, 1)
                = 0.5 * x * (1 + erf(x / sqrt(2)))

    Fast Tanh Approximation (Hendrycks & Gimpel, 2016):
        GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    """

    def __init__(self, approximate: bool = True) -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.approximate:
            # Tanh approximation (matches GPT-2 original implementation)
            sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
            inner = sqrt_2_over_pi * (x + 0.044715 * torch.pow(x, 3))
            return 0.5 * x * (1.0 + torch.tanh(inner))
        else:
            # Exact formulation using the error function (erf)
            return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class CustomReLU(nn.Module):
    """Rectified Linear Unit activation: ReLU(x) = max(0, x)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0.0)


def custom_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable Softmax function.

    Formula:
        softmax(z_i) = exp(z_i - max(z)) / sum_j exp(z_j - max(z))

    Subtracting max(z) guarantees exponents are <= 0, completely
    preventing catastrophic floating-point overflow (inf).
    """
    max_val, _ = torch.max(x, dim=dim, keepdim=True)
    exp_shifted = torch.exp(x - max_val)
    sum_exp = torch.sum(exp_shifted, dim=dim, keepdim=True)
    return exp_shifted / sum_exp

