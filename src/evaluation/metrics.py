"""Evaluation metrics for language models.

Implements:
1. Top-1 and Top-K Next-Token Prediction Accuracy
2. Cross-Entropy Loss and Perplexity (PPL)
3. Bits-Per-Token / Bits-Per-Character (BPC)
4. Distinct-N Generation Diversity Ratios
5. Detailed Parameter Inventory and Memory Footprint Analysis
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Union
import torch
import torch.nn as nn

from src.training.loss import compute_perplexity


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    top_k: int = 1,
    ignore_index: int = -100,
) -> float:
    """Computes Top-K next-token prediction accuracy.

    Formula:
        Acc_k = (1 / |Valid|) * sum_{i in Valid} 1(target_i in TopK(logits_i))

    Args:
        logits: Unnormalized predictions of shape (B, T, V) or (N, V)
        targets: Ground truth indices of shape (B, T) or (N,)
        top_k: Cutoff rank k (e.g. 1 for exact match, 5 for top-5)
        ignore_index: Target index to exclude from accuracy calculation

    Returns:
        Accuracy score in range [0.0, 1.0]
    """
    if logits.ndim == 3:
        B, T, V = logits.shape
        logits = logits.reshape(B * T, V)
        targets = targets.reshape(B * T)

    valid_mask = targets != ignore_index
    if not valid_mask.any():
        return 0.0

    valid_logits = logits[valid_mask]
    valid_targets = targets[valid_mask]

    if top_k == 1:
        preds = torch.argmax(valid_logits, dim=-1)
        correct = (preds == valid_targets).float().sum()
    else:
        k = min(top_k, valid_logits.size(-1))
        _, top_indices = torch.topk(valid_logits, k=k, dim=-1)
        # Check if true target appears along the candidate dimension
        correct = (top_indices == valid_targets.unsqueeze(1)).any(dim=-1).float().sum()

    return (correct / valid_targets.numel()).item()


def compute_bits_per_token(loss: Union[float, torch.Tensor]) -> float:
    """Converts natural log cross-entropy loss into base-2 Bits-Per-Token (BPC/BPT).

    Formula:
        BPT = Loss_nats / ln(2) = log_2(Perplexity)
    Represents the theoretical compression efficiency in bits per token.
    """
    if isinstance(loss, torch.Tensor):
        val = loss.item()
    else:
        val = float(loss)
    return val / math.log(2.0)


def compute_distinct_ngrams(text_or_tokens: Union[str, List[str]], n: int = 1) -> float:
    """Computes Distinct-N diversity ratio (Li et al., 2016).

    Formula:
        Distinct-N = Count(Unique N-grams) / Count(Total N-grams)

    A score near 1.0 indicates rich diversity, whereas a low score indicates
    repetitive degenerations.
    """
    if isinstance(text_or_tokens, str):
        # Character-level or word-level tokens
        tokens = list(text_or_tokens) if len(text_or_tokens.split()) <= 1 else text_or_tokens.split()
    else:
        tokens = text_or_tokens

    if len(tokens) < n:
        return 0.0

    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0

    return len(set(ngrams)) / len(ngrams)


def compute_parameter_breakdown(model: nn.Module) -> Dict[str, Any]:
    """Generates an itemized breakdown of model parameters and memory footprints."""
    category_counts: Dict[str, int] = defaultdict(int)
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        count = param.numel()
        total_params += count
        if param.requires_grad:
            trainable_params += count

        # Categorize parameter role
        if "wte" in name or "wpe" in name or "embeddings" in name:
            category_counts["embeddings"] += count
        elif "attn" in name or "c_attn" in name or "c_proj" in name and "mlp" not in name:
            category_counts["attention"] += count
        elif "mlp" in name or "c_fc" in name:
            category_counts["feedforward"] += count
        elif "ln" in name:
            category_counts["layernorm"] += count
        elif "lm_head" in name:
            category_counts["lm_head"] += count
        else:
            category_counts["other"] += count

    # Memory calculations
    bytes_fp32 = total_params * 4
    bytes_fp16 = total_params * 2
    bytes_int8 = total_params * 1

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "categories": dict(category_counts),
        "memory_fp32_mb": round(bytes_fp32 / (1024 * 1024), 3),
        "memory_fp16_mb": round(bytes_fp16 / (1024 * 1024), 3),
        "memory_int8_mb": round(bytes_int8 / (1024 * 1024), 3),
    }

