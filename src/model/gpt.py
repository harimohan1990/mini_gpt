"""Mini GPT: Complete Decoder-Only Generative Pre-trained Transformer.

Assembles:
1. Token and Positional Embeddings
2. Stack of N Pre-LayerNorm Transformer Blocks
3. Final Layer Normalization
4. Language Modeling Head (with optional weight tying)
5. Cross-Entropy Loss computation for training
6. Autoregressive text generation engine with sampling controls
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.embeddings import GPTEmbeddings
from src.model.transformer_block import CustomLayerNorm, TransformerBlock
from src.training.loss import custom_cross_entropy


@dataclass
class GPTConfig:
    """Hyperparameters configuring a Mini GPT model instance."""
    vocab_size: int = 70
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True
    tie_weights: bool = True


class MiniGPT(nn.Module):
    """Complete Decoder-Only Generative Pre-trained Transformer."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # 1. Input Embeddings (Token + Learned Positional)
        self.transformer = nn.ModuleDict(
            dict(
                embeddings=GPTEmbeddings(
                    vocab_size=config.vocab_size,
                    block_size=config.block_size,
                    n_embd=config.n_embd,
                    dropout=config.dropout,
                ),
                # 2. Stack of Transformer Decoder Blocks
                h=nn.ModuleList(
                    [
                        TransformerBlock(
                            n_embd=config.n_embd,
                            n_head=config.n_head,
                            block_size=config.block_size,
                            dropout=config.dropout,
                            bias=config.bias,
                        )
                        for _ in range(config.n_layer)
                    ]
                ),
                # 3. Final Layer Normalization
                ln_f=CustomLayerNorm(config.n_embd, bias=config.bias),
            )
        )

        # 4. Language Modeling Head: projects n_embd back to vocab_size logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: tie token embedding weights to output projection weights
        if config.tie_weights:
            self.lm_head.weight = self.transformer.embeddings.wte.weight

        # Initialize all model weights according to GPT-2 standard
        self.apply(self._init_weights)

        # Apply special scaled initialization to residual projections (c_proj)
        # to prevent variance from growing proportionally with depth
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Returns the total number of trainable parameters in the model.

        If non_embedding is True, subtracts positional and token embedding counts.
        """
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n_params -= self.transformer.embeddings.wpe.weight.numel()
            if not self.config.tie_weights:
                n_params -= self.transformer.embeddings.wte.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of Mini GPT.

        Args:
            idx: Tensor of shape (B, T) containing token IDs
            targets: Optional tensor of shape (B, T) containing target next-token IDs

        Returns:
            logits: Output classification scores of shape (B, T, vocab_size)
            loss: Cross-entropy scalar loss if targets are provided, else None
        """
        device = idx.device
        B, T = idx.shape

        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds maximum configured context window {self.config.block_size}"
            )

        # 1. Forward through Embeddings: (B, T) -> (B, T, n_embd)
        x = self.transformer.embeddings(idx)

        # 2. Forward through stack of N Transformer Blocks: (B, T, n_embd)
        for block in self.transformer.h:
            x, _ = block(x)

        # 3. Final Layer Normalization: (B, T, n_embd)
        x = self.transformer.ln_f(x)

        # 4. Language Modeling Head: (B, T, n_embd) -> (B, T, vocab_size)
        logits = self.lm_head(x)

        # 5. Compute loss if supervision targets are supplied
        loss = None
        if targets is not None:
            loss = custom_cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressively generates new tokens conditioned on prompt idx.

        Args:
            idx: Tensor of shape (B, T) containing initial prompt token IDs
            max_new_tokens: Number of continuation tokens to sample
            temperature: Sampling temperature (higher = more diverse, lower = more deterministic)
            top_k: Optional top-k filtering truncation
            eos_token_id: Optional token ID to halt generation early upon emission

        Returns:
            Tensor of shape (B, T + max_new_tokens) containing full generated sequence
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop prompt to context window if it exceeds block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]

            # Forward pass: get logits for current context
            logits, _ = self(idx_cond)

            # Pluck logits at the final time step: shape (B, vocab_size)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Optional Top-K truncation
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Softmax to obtain probability distribution
            probs = F.softmax(logits, dim=-1)

            # Sample next token from the multinomial distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append sampled token to running sequence
            idx = torch.cat((idx, idx_next), dim=1)

            # Stop condition if EOS token is generated
            if eos_token_id is not None and (idx_next == eos_token_id).all():
                break

        return idx

