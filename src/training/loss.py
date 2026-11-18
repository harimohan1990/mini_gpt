"""Loss functions and evaluation metrics for Mini GPT."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


def custom_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Numerically stable Cross-Entropy Loss for language modeling.

    Mathematically:
        Given unnormalized logits z in R^V and ground-truth index y in {0, ..., V-1}:
        P(y) = exp(z_y) / sum_{j=1}^V exp(z_j)
        Loss = -log P(y) = -z_y + log sum_{j=1}^V exp(z_j)

    Using the Log-Sum-Exp trick to prevent numeric overflow:
        m = max_j(z_j)
        log sum_j exp(z_j) = m + log sum_j exp(z_j - m)

    Supports:
        - 2D logits (N, V) with 1D targets (N,)
        - 3D logits (B, T, V) with 2D targets (B, T)
    """
    if logits.ndim == 3:
        # Flatten batch and sequence dimensions: (B, T, V) -> (B*T, V)
        B, T, V = logits.shape
        logits = logits.view(B * T, V)
        targets = targets.view(B * T)

    N, V = logits.shape

    # 1. Numerically stable Log-Sum-Exp
    max_logits, _ = torch.max(logits, dim=-1, keepdim=True)  # (N, 1)
    shifted_logits = logits - max_logits
    log_sum_exp = max_logits.squeeze(-1) + torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))  # (N,)

    # 2. Extract logits corresponding to ground truth target labels: z_y
    # Mask out ignore_index (e.g. padding tokens)
    valid_mask = targets != ignore_index
    safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))

    # Gather target logits: logits[i, targets[i]]
    target_logits = torch.gather(logits, dim=1, index=safe_targets.unsqueeze(1)).squeeze(1)  # (N,)

    # Loss = -z_y + log_sum_exp
    per_token_loss = -target_logits + log_sum_exp

    # Zero out ignored positions
    per_token_loss = per_token_loss * valid_mask.float()

    if reduction == "mean":
        num_valid = valid_mask.float().sum()
        return per_token_loss.sum() / torch.clamp(num_valid, min=1.0)
    elif reduction == "sum":
        return per_token_loss.sum()
    elif reduction == "none":
        return per_token_loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


class CustomCrossEntropyLoss(nn.Module):
    """Module wrapper for custom cross entropy loss."""

    def __init__(self, ignore_index: int = -100, reduction: str = "mean") -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return custom_cross_entropy(
            logits=logits,
            targets=targets,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


def compute_perplexity(loss: float | torch.Tensor) -> float:
    """Computes Perplexity (PPL) from cross entropy loss.

    Formula:
        Perplexity = exp(Loss)

    Represents the geometric mean of the inverse probability assigned to the
    target tokens. Intuitively, it measures the effective vocabulary size the
    model is choosing from at each step.
    """
    if isinstance(loss, torch.Tensor):
        loss_val = loss.item()
    else:
        loss_val = float(loss)

    try:
        return math.exp(loss_val)
    except OverflowError:
        return float("inf")

