"""Optimizer implementations and parameter group configurations for Mini GPT."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class CustomAdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay (Loshchilov & Hutter, 2017).

    Mathematical Formulation at step t:
        1. Weight Decay:
           theta_t = theta_{t-1} - lr * weight_decay * theta_{t-1}

        2. Gradient Moments:
           m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
           v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t)^2

        3. Bias Correction:
           m_hat_t = m_t / (1 - beta1^t)
           v_hat_t = v_t / (1 - beta2^t)

        4. Parameter Update:
           theta_t = theta_t - lr * m_hat_t / (sqrt(v_hat_t) + eps)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("CustomAdamW does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values (m_t)
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values (v_t)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # 1. Decoupled weight decay
                if weight_decay > 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                # 2. Update biased 1st and 2nd moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 3. Compute bias corrections
                bias_correction1 = 1.0 - beta1**t
                bias_correction2 = 1.0 - beta2**t

                # Step size adjustment for bias correction
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # 4. Apply parameter update
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def configure_optimizers(
    model: nn.Module,
    weight_decay: float = 0.01,
    learning_rate: float = 3e-4,
    betas: Tuple[float, float] = (0.9, 0.95),
    device: str = "cpu",
) -> torch.optim.Optimizer:
    """Configures parameter groups separating decayable weights from non-decayable parameters.

    In modern Transformer language models:
    - 2D parameters (weights of Linear and Embedding layers) undergo weight decay.
    - 1D parameters (biases, LayerNorm scale and shift) do NOT undergo weight decay.

    Applying weight decay to biases and LayerNorm parameters degrades training stability.
    """
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Any parameter that is 2D or greater gets weight decayed (e.g. weight matrices)
        # Any 1D parameter (biases, layernorm gamma/beta) does not
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = CustomAdamW(
        optim_groups,
        lr=learning_rate,
        betas=betas,
    )
    return optimizer

