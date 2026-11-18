"""Token and Positional Embeddings for Mini GPT.

Implements discrete token embeddings combined with learned positional embeddings,
providing the model with both semantic content and sequence order information.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GPTEmbeddings(nn.Module):
    """Combines Token Embeddings (wte) and Learned Positional Embeddings (wpe).

    Mathematical Formulation:
        Given token sequence idx in {0, ..., V-1}^(B x T):
        Token Embeddings:       E_tok = W_te[idx]       in R^(B x T x d_model)
        Positional Indices:     pos   = [0, 1, ..., T-1] in {0, ..., T_max - 1}^T
        Positional Embeddings:  E_pos = W_pe[pos]       in R^(T x d_model)
        Combined Representation:
            E = Dropout(E_tok + E_pos)                   in R^(B x T x d_model)
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd

        # Token embedding lookup table: W_te in R^(vocab_size x n_embd)
        self.wte = nn.Embedding(vocab_size, n_embd)

        # Learned positional embedding table: W_pe in R^(block_size x n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)

        # Embedding dropout
        self.drop = nn.Dropout(dropout)

        # Initialize weights with standard deviation 0.02
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.wte.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.wpe.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Args:
            idx: Tensor of shape (B, T) containing token IDs.

        Returns:
            Tensor of shape (B, T, n_embd) containing injected representations.
        """
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {T}, context window block_size is {self.block_size}"
            )

        # Position indices: [0, 1, ..., T - 1] on the same device as inputs
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T,)

        # Token and positional embeddings
        tok_emb = self.wte(idx)  # (B, T, n_embd)
        pos_emb = self.wpe(pos)  # (T, n_embd) -> automatically broadcasts along batch dimension

        # Injected sum + dropout
        x = self.drop(tok_emb + pos_emb)  # (B, T, n_embd)
        return x

