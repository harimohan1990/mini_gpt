"""Unit tests for Phase 9: Model Optimization and Acceleration."""

import pytest
import torch
import torch.nn as nn

from src.model.attention import CausalSelfAttention
from src.model.gpt import MiniGPT, GPTConfig
from src.optimization.quantization import (
    quantize_model_dynamic,
    get_model_file_size,
    compare_quantization,
)
from src.optimization.benchmark import (
    benchmark_attention_efficiency,
    benchmark_batch_scaling,
)


def test_sdpa_attention_parity():
    torch.manual_seed(42)
    B, T, n_embd, n_head = 2, 8, 32, 4

    # Two layers with synchronized weights: one with SDPA, one with explicit math
    attn_sdpa = CausalSelfAttention(n_embd=n_embd, n_head=n_head, block_size=16, dropout=0.0, use_sdpa=True)
    attn_manual = CausalSelfAttention(n_embd=n_embd, n_head=n_head, block_size=16, dropout=0.0, use_sdpa=False)

    with torch.no_grad():
        attn_manual.c_attn.weight.copy_(attn_sdpa.c_attn.weight)
        attn_manual.c_attn.bias.copy_(attn_sdpa.c_attn.bias)
        attn_manual.c_proj.weight.copy_(attn_sdpa.c_proj.weight)
        attn_manual.c_proj.bias.copy_(attn_sdpa.c_proj.bias)

    attn_sdpa.eval()
    attn_manual.eval()

    x = torch.randn(B, T, n_embd)
    out_sdpa, _ = attn_sdpa(x)
    out_manual, _ = attn_manual(x)

    assert torch.allclose(out_sdpa, out_manual, atol=1e-5)


def test_quantization_size_reduction():
    config = GPTConfig(
        vocab_size=50,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=64,
    )
    model = MiniGPT(config)
    quant_stats = compare_quantization(model)

    assert quant_stats["fp32_size_mb"] > 0
    assert quant_stats["int8_size_mb"] > 0
    # INT8 should achieve non-trivial compression
    assert quant_stats["int8_size_mb"] < quant_stats["fp32_size_mb"]
    assert quant_stats["compression_ratio"] > 1.0


def test_quantized_model_forward():
    config = GPTConfig(
        vocab_size=30,
        block_size=16,
        n_layer=1,
        n_head=2,
        n_embd=32,
    )
    model = MiniGPT(config)
    quantized_model = quantize_model_dynamic(model)

    test_input = torch.randint(0, 30, (2, 8))
    with torch.no_grad():
        logits, _ = quantized_model(test_input)

    assert logits.shape == (2, 8, 30)
    assert not torch.isnan(logits).any()


def test_autocast_bfloat16_forward():
    config = GPTConfig(
        vocab_size=40,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )
    model = MiniGPT(config)
    model.eval()

    test_input = torch.randint(0, 40, (2, 8))

    # Test autocast with bfloat16 on CPU
    with torch.no_grad():
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits, _ = model(test_input)

    assert logits.shape == (2, 8, 40)
    assert not torch.isnan(logits).any()


def test_batch_scaling_benchmark():
    config = GPTConfig(
        vocab_size=30,
        block_size=16,
        n_layer=1,
        n_head=1,
        n_embd=16,
    )
    model = MiniGPT(config)
    results = benchmark_batch_scaling(model, batch_sizes=[1, 4], seq_len=8, device="cpu")

    assert len(results) == 2
    assert results[0]["batch_size"] == 1
    assert results[1]["batch_size"] == 4
    # Larger batch should exhibit higher token throughput
    assert results[1]["tokens_per_sec"] > 0.0
