"""Sampling strategies and logit processors for Mini GPT generation.

Implements:
1. Temperature Scaling
2. Top-K Truncation Filtering
3. Top-P (Nucleus) Cumulative Probability Filtering
4. Repetition Penalty
5. Greedy Selection and Categorical Multinomial Sampling
"""

from __future__ import annotations

from typing import List, Optional, Union
import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Scales logits by inverse temperature.

    Mathematically:
        z'_i = z_i / T

    - T -> 0: Point-mass distribution (approaches greedy argmax).
    - T = 1: Unaltered model confidence.
    - T > 1: Flattens distribution towards uniform (higher entropy/creativity).
    """
    if temperature <= 0.0:
        raise ValueError(f"Temperature must be strictly positive, got {temperature}")
    return logits / temperature


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Truncates logits retaining only the top K highest scoring candidates.

    All tokens with rank > K are masked with -infinity, ensuring zero probability
    after softmax.
    """
    if top_k <= 0:
        return logits

    k = min(top_k, logits.size(-1))
    # Extract top k values
    top_values, _ = torch.topk(logits, k, dim=-1)
    # The k-th highest value for each sequence in the batch
    kth_value = top_values[..., [-1]]
    # Mask all logits lower than the k-th threshold
    return torch.where(logits < kth_value, torch.tensor(float("-inf"), device=logits.device), logits)


def apply_top_p(logits: torch.Tensor, top_p: float, min_tokens_to_keep: int = 1) -> torch.Tensor:
    """Nucleus (Top-P) Filtering (Holtzman et al., 2019).

    Dynamically selects the smallest set of high-probability tokens whose cumulative
    probability exceeds threshold P. Tokens outside this nucleus are masked with -infinity.

    Unlike Top-K with a fixed cutoff, Top-P adapts its candidate pool width depending
    on whether the model is certain (few tokens) or uncertain (many tokens).
    """
    if not (0.0 < top_p <= 1.0):
        return logits

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    # Compute cumulative probability mass
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Identify tokens exceeding the cumulative threshold P
    # Shift right so the first token exceeding threshold is preserved
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False  # Keep at least the top-1 token

    if min_tokens_to_keep > 1:
        sorted_indices_to_remove[..., :min_tokens_to_keep] = False

    # Scatter mask back to original logit indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove,
    )
    return logits.masked_fill(indices_to_remove, float("-inf"))


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: Union[torch.Tensor, List[int]],
    penalty: float = 1.0,
) -> torch.Tensor:
    """Applies repetition penalty to logits of previously generated tokens (Keskar et al., 2019).

    Formula for token ID i that has appeared in context:
        If z_i > 0:  z_i <- z_i / penalty
        If z_i <= 0: z_i <- z_i * penalty
    Where penalty >= 1.0. A penalty of 1.0 is identity (no effect).
    """
    if penalty == 1.0:
        return logits

    if isinstance(generated_tokens, torch.Tensor):
        unique_tokens = torch.unique(generated_tokens)
    else:
        unique_tokens = torch.tensor(list(set(generated_tokens)), dtype=torch.long, device=logits.device)

    # Clone logits to avoid in-place mutation
    logits_mod = logits.clone()

    for token_id in unique_tokens:
        if token_id < 0 or token_id >= logits_mod.size(-1):
            continue
        val = logits_mod[..., token_id]
        # Positive logits get divided; negative logits get multiplied (pushed further negative)
        logits_mod[..., token_id] = torch.where(val > 0, val / penalty, val * penalty)

    return logits_mod


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """Processes logits and draws the next token index via greedy or multinomial sampling.

    Args:
        logits: Unnormalized logits of shape (B, V) for the last token position
        temperature: Sampling temperature
        top_k: Optional integer cutoff
        top_p: Optional float cumulative mass cutoff

    Returns:
        Tensor of shape (B, 1) containing sampled token IDs.
    """
    # Deterministic greedy selection if temperature approaches zero
    if temperature < 1e-4:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 1. Temperature scaling
    scaled_logits = apply_temperature(logits, temperature=temperature)

    # 2. Top-K filtering
    if top_k is not None and top_k > 0:
        scaled_logits = apply_top_k(scaled_logits, top_k=top_k)

    # 3. Top-P (Nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        scaled_logits = apply_top_p(scaled_logits, top_p=top_p)

    # 4. Softmax probability distribution
    probs = F.softmax(scaled_logits, dim=-1)

    # 5. Draw sample from categorical distribution
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

