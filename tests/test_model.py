"""Unit tests for Phase 4: Full GPT Architecture."""

import pytest
import torch
import torch.nn as nn

from src.model.embeddings import GPTEmbeddings
from src.model.attention import CausalSelfAttention
from src.model.feedforward import FeedForward
from src.model.transformer_block import TransformerBlock
from src.model.gpt import MiniGPT, GPTConfig


def test_gpt_embeddings_shapes_and_limits():
    vocab_size, block_size, n_embd = 50, 16, 32
    embeddings = GPTEmbeddings(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd)

    # Valid sequence length
    idx = torch.randint(0, vocab_size, (2, 8))
    out = embeddings(idx)
    assert out.shape == (2, 8, n_embd)

    # Exceeding block_size must raise ValueError
    invalid_idx = torch.randint(0, vocab_size, (1, block_size + 1))
    with pytest.raises(ValueError):
        _ = embeddings(invalid_idx)


def test_causal_self_attention_masking():
    torch.manual_seed(42)
    B, T, n_embd, n_head = 2, 6, 32, 4
    attn_layer = CausalSelfAttention(n_embd=n_embd, n_head=n_head, block_size=16, dropout=0.0)
    attn_layer.eval()

    x = torch.randn(B, T, n_embd)
    out, attn_weights = attn_layer(x, return_attention=True)

    assert out.shape == (B, T, n_embd)
    assert attn_weights is not None
    assert attn_weights.shape == (B, n_head, T, T)

    # Verify causal mask: for any position i, attention to j > i must be strictly zero!
    for i in range(T):
        for j in range(i + 1, T):
            # Attention weight must be 0.0 for future tokens
            assert torch.all(attn_weights[:, :, i, j] == 0.0)

    # Verify absence of lookahead leakage:
    # Modifying the last token x[:, -1, :] MUST NOT affect the output at earlier positions out[:, :-1, :]
    x_modified = x.clone()
    x_modified[:, -1, :] += 50.0  # Drastic change to the final token

    out_modified, _ = attn_layer(x_modified, return_attention=False)
    # The first T-1 positions must remain identical
    assert torch.allclose(out[:, :-1, :], out_modified[:, :-1, :], atol=1e-5)


def test_feedforward_shape_and_flow():
    B, T, n_embd = 2, 8, 32
    ff = FeedForward(n_embd=n_embd, dropout=0.0)
    x = torch.randn(B, T, n_embd)
    y = ff(x)
    assert y.shape == (B, T, n_embd)


def test_transformer_block_residual_flow():
    B, T, n_embd, n_head = 2, 8, 32, 4
    block = TransformerBlock(n_embd=n_embd, n_head=n_head, block_size=16, dropout=0.0)
    x = torch.randn(B, T, n_embd, requires_grad=True)

    y, _ = block(x)
    assert y.shape == (B, T, n_embd)

    # Verify gradient flow back through residual connections
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_mini_gpt_forward_and_loss():
    config = GPTConfig(
        vocab_size=60,
        block_size=16,
        n_layer=2,
        n_head=4,
        n_embd=32,
        dropout=0.0,
        tie_weights=True,
    )
    model = MiniGPT(config)
    model.eval()

    B, T = 2, 8
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    # Inference forward pass (targets=None)
    logits, loss_none = model(idx)
    assert logits.shape == (B, T, config.vocab_size)
    assert loss_none is None

    # Training forward pass (targets provided)
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (B, T, config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0  # Scalar loss
    assert loss.item() > 0.0

    # Verify weight tying
    assert model.lm_head.weight is model.transformer.embeddings.wte.weight


def test_mini_gpt_parameter_count():
    config = GPTConfig(
        vocab_size=70,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=64,
        tie_weights=True,
    )
    model = MiniGPT(config)
    num_params = model.get_num_params(non_embedding=False)
    assert num_params > 0
    # Parameter count should be well under 1 million for this tiny config
    assert num_params < 500_000


def test_mini_gpt_autoregressive_generation():
    config = GPTConfig(
        vocab_size=50,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )
    model = MiniGPT(config)
    model.eval()

    prompt = torch.tensor([[1, 2, 3]])  # B=1, T=3
    max_new_tokens = 5
    generated = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=1.0, top_k=5)

    assert generated.shape == (1, 3 + max_new_tokens)
    # The beginning of the generated sequence must strictly preserve the prompt
    assert torch.equal(generated[:, :3], prompt)

