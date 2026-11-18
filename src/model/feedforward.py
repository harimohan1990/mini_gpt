"""Feed-Forward Network (MLP) for Mini GPT.

Applies position-wise non-linear feature transformation to token representations.
While Self-Attention mixes information across sequence tokens, the Feed-Forward
network processes and refines features within each token representation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from src.model.foundations import CustomGELU


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network (MLP).

    Mathematical Formulation:
        Given x in R^(B x T x d_model):
        1. Linear Expansion:  h = x @ W_1.T + b_1     in R^(B x T x 4*d_model)
        2. Non-Linear Gating: a = GELU(h)             in R^(B x T x 4*d_model)
        3. Linear Projection: y = a @ W_2.T + b_2     in R^(B x T x d_model)
        4. Regularization:    Out = Dropout(y)        in R^(B x T x d_model)
    """

    def __init__(
        self,
        n_embd: int,
        dropout: float = 0.1,
        bias: bool = True,
        expansion_factor: int = 4,
    ) -> None:
        super().__init__()
        self.n_embd = n_embd
        d_ff = expansion_factor * n_embd

        # 1. Expand dimension from n_embd to 4 * n_embd
        self.c_fc = nn.Linear(n_embd, d_ff, bias=bias)

        # 2. GELU non-linearity (with tanh approximation matching GPT-2)
        self.act = CustomGELU(approximate=True)

        # 3. Project back from 4 * n_embd to n_embd
        self.c_proj = nn.Linear(d_ff, n_embd, bias=bias)

        # 4. Residual dropout
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.c_fc.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02)
        if self.c_fc.bias is not None:
            nn.init.zeros_(self.c_fc.bias)
        if self.c_proj.bias is not None:
            nn.init.zeros_(self.c_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Tensor of shape (B, T, n_embd)

        Returns:
            Tensor of shape (B, T, n_embd)
        """
        # (B, T, n_embd) -> (B, T, 4 * n_embd)
        h = self.c_fc(x)
        # Activation
        h = self.act(h)
        # (B, T, 4 * n_embd) -> (B, T, n_embd)
        y = self.c_proj(h)
        # Dropout
        return self.dropout(y)

