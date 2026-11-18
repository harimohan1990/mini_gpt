"""Transformer Decoder Block for Mini GPT.

Implements the Pre-LayerNorm Transformer Block combining Multi-Head Causal
Self-Attention, position-wise Feed-Forward Networks, and additive residual highways.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.model.attention import CausalSelfAttention
from src.model.feedforward import FeedForward


class CustomLayerNorm(nn.Module):
    """Layer Normalization over the final channel dimension (d_model).

    Formula:
        y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
    where gamma (weight) and beta (bias) are learnable parameters.
    """

    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mean and variance along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        out = x_norm * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer Decoder Block.

    Mathematical Formulation:
        Given input x in R^(B x T x d_model):
        1. Attention Sub-layer with Pre-LN:
           a = x + Attention(LN_1(x))
        2. Feed-Forward Sub-layer with Pre-LN:
           out = a + FeedForward(LN_2(a))

    The additive residual connections ('+') ensure gradient signals flow
    unhindered from output back to input across arbitrarily deep networks.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        # Pre-attention Layer Normalization
        self.ln_1 = CustomLayerNorm(n_embd, bias=bias)
        # Causal Multi-Head Self-Attention
        self.attn = CausalSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout,
            bias=bias,
        )
        # Pre-MLP Layer Normalization
        self.ln_2 = CustomLayerNorm(n_embd, bias=bias)
        # Position-wise Feed-Forward MLP
        self.mlp = FeedForward(
            n_embd=n_embd,
            dropout=dropout,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Args:
            x: Input representations of shape (B, T, n_embd)
            return_attention: Whether to return self-attention weights

        Returns:
            Tuple of:
                x: Transformed representations of shape (B, T, n_embd)
                attn_weights: Attention weights of shape (B, H, T, T) if requested
        """
        # 1. Residual Self-Attention with Pre-LayerNorm
        attn_out, attn_weights = self.attn(self.ln_1(x), return_attention=return_attention)
        x = x + attn_out

        # 2. Residual Feed-Forward MLP with Pre-LayerNorm
        x = x + self.mlp(self.ln_2(x))

        return x, attn_weights

