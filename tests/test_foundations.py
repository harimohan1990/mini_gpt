"""Unit tests for Phase 3: Neural Network Foundations."""

import math
import pytest
import torch
import torch.nn as nn

from src.model.foundations import (
    CustomLinear,
    CustomEmbedding,
    CustomGELU,
    CustomReLU,
    custom_softmax,
)
from src.training.loss import (
    custom_cross_entropy,
    CustomCrossEntropyLoss,
    compute_perplexity,
)
from src.training.optimizer import CustomAdamW, configure_optimizers


def test_custom_linear_forward_and_backward():
    torch.manual_seed(42)
    B, T, d_in, d_out = 2, 4, 8, 16

    custom_linear = CustomLinear(d_in, d_out, bias=True)
    ref_linear = nn.Linear(d_in, d_out, bias=True)

    # Sync weights and biases
    with torch.no_grad():
        ref_linear.weight.copy_(custom_linear.weight)
        ref_linear.bias.copy_(custom_linear.bias)

    x = torch.randn(B, T, d_in, requires_grad=True)
    x_ref = x.clone().detach().requires_grad_(True)

    out_custom = custom_linear(x)
    out_ref = ref_linear(x_ref)

    assert out_custom.shape == (B, T, d_out)
    assert torch.allclose(out_custom, out_ref, atol=1e-6)

    # Backward pass gradient check
    loss_custom = out_custom.sum()
    loss_ref = out_ref.sum()

    loss_custom.backward()
    loss_ref.backward()

    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6)
    assert torch.allclose(custom_linear.weight.grad, ref_linear.weight.grad, atol=1e-6)
    assert torch.allclose(custom_linear.bias.grad, ref_linear.bias.grad, atol=1e-6)


def test_custom_embedding():
    torch.manual_seed(42)
    vocab_size, d_model = 100, 32
    B, T = 4, 8

    custom_emb = CustomEmbedding(vocab_size, d_model)
    ref_emb = nn.Embedding(vocab_size, d_model)

    with torch.no_grad():
        ref_emb.weight.copy_(custom_emb.weight)

    indices = torch.randint(0, vocab_size, (B, T))
    out_custom = custom_emb(indices)
    out_ref = ref_emb(indices)

    assert out_custom.shape == (B, T, d_model)
    assert torch.allclose(out_custom, out_ref, atol=1e-6)

    # Backward gradient flow
    loss_custom = out_custom.sum()
    loss_ref = out_ref.sum()
    loss_custom.backward()
    loss_ref.backward()

    assert torch.allclose(custom_emb.weight.grad, ref_emb.weight.grad, atol=1e-6)


def test_custom_gelu():
    x = torch.linspace(-3.0, 3.0, 100)
    gelu_approx = CustomGELU(approximate=True)
    ref_gelu = nn.GELU(approximate="tanh")

    out_custom = gelu_approx(x)
    out_ref = ref_gelu(x)

    assert torch.allclose(out_custom, out_ref, atol=1e-5)


def test_custom_softmax_numerical_stability():
    # Test large numbers that would cause exp() overflow if un-shifted
    x = torch.tensor([[1000.0, 1001.0, 1002.0], [-1000.0, 0.0, 10.0]])
    probs = custom_softmax(x, dim=-1)
    ref_probs = torch.softmax(x, dim=-1)

    assert not torch.isnan(probs).any()
    assert not torch.isinf(probs).any()
    assert torch.allclose(probs, ref_probs, atol=1e-6)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_custom_cross_entropy_3d_tensor():
    torch.manual_seed(42)
    B, T, V = 3, 5, 20
    logits = torch.randn(B, T, V, requires_grad=True)
    targets = torch.randint(0, V, (B, T))

    loss_custom = custom_cross_entropy(logits, targets)
    loss_ref = nn.functional.cross_entropy(logits.view(-1, V), targets.view(-1))

    assert torch.allclose(loss_custom, loss_ref, atol=1e-6)

    # Gradient check
    loss_custom.backward()
    grad_custom = logits.grad.clone()

    logits.grad.zero_()
    loss_ref = nn.functional.cross_entropy(logits.view(-1, V), targets.view(-1))
    loss_ref.backward()
    grad_ref = logits.grad.clone()

    assert torch.allclose(grad_custom, grad_ref, atol=1e-6)


def test_compute_perplexity():
    loss = torch.tensor(0.0)
    assert pytest.approx(compute_perplexity(loss), rel=1e-5) == 1.0

    loss = torch.tensor(math.log(10.0))
    assert pytest.approx(compute_perplexity(loss), rel=1e-4) == 10.0


def test_custom_adamw_step_parity():
    torch.manual_seed(42)
    p_custom = nn.Parameter(torch.randn(10, 10))
    p_ref = nn.Parameter(p_custom.clone().detach())

    opt_custom = CustomAdamW([p_custom], lr=1e-3, weight_decay=0.01)
    opt_ref = torch.optim.AdamW([p_ref], lr=1e-3, weight_decay=0.01)

    for _ in range(5):
        # Fake loss and gradients
        loss_custom = (p_custom ** 2).sum()
        loss_ref = (p_ref ** 2).sum()

        loss_custom.backward()
        loss_ref.backward()

        opt_custom.step()
        opt_ref.step()

        opt_custom.zero_grad()
        opt_ref.zero_grad()

        assert torch.allclose(p_custom, p_ref, atol=1e-6)


def test_configure_optimizers_parameter_segregation():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = CustomLinear(10, 20, bias=True)
            self.ln = nn.LayerNorm(20)

    model = DummyModel()
    optimizer = configure_optimizers(model, weight_decay=0.1)

    # Decay group should contain linear.weight (2D)
    decay_params = optimizer.param_groups[0]["params"]
    # No decay group should contain linear.bias (1D), ln.weight (1D), ln.bias (1D)
    no_decay_params = optimizer.param_groups[1]["params"]

    assert len(decay_params) == 1  # linear.weight
    assert len(no_decay_params) == 3  # linear.bias, ln.weight, ln.bias
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 0.0

