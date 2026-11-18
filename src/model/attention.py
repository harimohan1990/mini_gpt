"""Causal Multi-Head Self-Attention for Mini GPT.

Implements:
1. Fused Query, Key, Value linear projections
2. Scaled Dot-Product Attention
3. Causal autoregressive masking
4. Multi-head splitting and recombination
5. Output projection and residual dropout
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention mechanism.

    Mathematical Formulation:
        Let X in R^(B x T x d_model).
        1. Q, K, V = X W_qkv                               in R^(B x T x 3*d_model)
        2. Reshape into H heads of dim d_k = d_model / H:
           Q, K, V in R^(B x H x T x d_k)
        3. Scaled Dot-Product:
           Scores = (Q @ K.T) / sqrt(d_k)                  in R^(B x H x T x T)
        4. Causal Masking:
           Scores_masked = Scores.masked_fill(mask == 0, -inf)
        5. Attention Probabilities:
           Attn = Softmax(Scores_masked, dim=-1)           in R^(B x H x T x T)
        6. Value Aggregation:
           Out_heads = Attn @ V                            in R^(B x H x T x d_k)
        7. Recombination & Output Projection:
           Out = Out_heads.view(B, T, d_model) @ W_proj    in R^(B x T x d_model)
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.1,
        bias: bool = True,
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size
        self.dropout_p = dropout
        self.use_sdpa = use_sdpa and hasattr(F, "scaled_dot_product_attention")

        # Key, Query, Value projections in a single batch-fused linear projection
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)

        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        # Regularization dropouts
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Lower-triangular causal mask: mask[i, j] = 1 if j <= i else 0
        # Registered as buffer so it moves with model device without being an optimizer parameter
        self.register_buffer(
            "bias_mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.c_attn.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02)
        if self.c_attn.bias is not None:
            nn.init.zeros_(self.c_attn.bias)
        if self.c_proj.bias is not None:
            nn.init.zeros_(self.c_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Args:
            x: Input activations of shape (B, T, n_embd)
            return_attention: If True, also returns attention weights

        Returns:
            Tuple of:
                y: Output activations of shape (B, T, n_embd)
                attn_weights: Attention weights of shape (B, H, T, T) if requested
        """
        B, T, C = x.shape

        # 1. Calculate Q, K, V projections for all heads in batch
        # c_attn produces (B, T, 3 * C)
        qkv = self.c_attn(x)
        # Split into Query, Key, and Value: each is (B, T, C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 2. Reshape and transpose for multi-head attention:
        # (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        if self.use_sdpa and not return_attention:
            # PyTorch 2.0+ FlashAttention / Memory-Efficient Kernel
            drop_rate = self.dropout_p if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=drop_rate,
                is_causal=True,
            )
            attn = None
        else:
            # Explicit educational math path
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            scores = scores.masked_fill(self.bias_mask[:, :, :T, :T] == 0, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            attn_dropped = self.attn_dropout(attn)
            out = torch.matmul(attn_dropped, v)

        # 4. Re-assemble all head outputs side by side:
        # (B, H, T, head_dim) -> (B, T, H, head_dim) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 5. Output projection and residual dropout
        y = self.resid_dropout(self.c_proj(out))

        return (y, attn if return_attention else None)

