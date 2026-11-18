"""Benchmarking and profiling suite for Mini GPT optimizations.

Measures:
1. Device execution latency (CPU vs MPS vs CUDA)
2. Attention efficiency: Standard Attention vs PyTorch SDPA
3. Quantization latency: FP32 vs Dynamic INT8
4. Batch size throughput scaling
5. Time-To-First-Token (TTFT) and Inter-Token Latency (ITL)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.gpt import MiniGPT, GPTConfig
from src.optimization.quantization import quantize_model_dynamic


def benchmark_device_inference(
    model: MiniGPT,
    device: str = "cpu",
    seq_len: int = 64,
    num_runs: int = 25,
    warmup: int = 5,
) -> Dict[str, float]:
    """Profiles forward pass latency and throughput on a specified hardware device."""
    model.to(device)
    model.eval()

    dummy_input = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)

    # Warmup runs to stabilize caches and kernel compilation
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    latencies: List[float] = []

    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()

            t0 = time.perf_counter()
            _ = model(dummy_input)

            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()

            dt = (time.perf_counter() - t0) * 1000.0  # ms
            latencies.append(dt)

    mean_latency = sum(latencies) / len(latencies)
    throughput = (seq_len / (mean_latency / 1000.0))

    return {
        "device": device,
        "mean_latency_ms": round(mean_latency, 2),
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "throughput_tokens_per_sec": round(throughput, 1),
    }


def benchmark_attention_efficiency(
    n_embd: int = 128,
    n_head: int = 4,
    seq_len: int = 128,
    batch_size: int = 16,
    num_runs: int = 30,
) -> Dict[str, Any]:
    """Compares execution time of standard manual attention vs PyTorch SDPA."""
    device = "cpu"
    B, H, T, D = batch_size, n_head, seq_len, n_embd // n_head

    q = torch.randn(B, H, T, D, device=device)
    k = torch.randn(B, H, T, D, device=device)
    v = torch.randn(B, H, T, D, device=device)
    mask = torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)

    # 1. Benchmark Standard Manual Attention
    manual_times: List[float] = []
    scale = 1.0 / (D ** 0.5)

    for _ in range(num_runs):
        t0 = time.perf_counter()
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        _ = torch.matmul(attn, v)
        manual_times.append((time.perf_counter() - t0) * 1000.0)

    # 2. Benchmark PyTorch SDPA (Scaled Dot-Product Attention)
    sdpa_times: List[float] = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        sdpa_times.append((time.perf_counter() - t0) * 1000.0)

    mean_manual = sum(manual_times) / len(manual_times)
    mean_sdpa = sum(sdpa_times) / len(sdpa_times)
    speedup = round(mean_manual / max(mean_sdpa, 1e-6), 2)

    return {
        "manual_attention_ms": round(mean_manual, 3),
        "sdpa_attention_ms": round(mean_sdpa, 3),
        "speedup_factor": speedup,
    }


def benchmark_quantization_latency(
    model: MiniGPT,
    seq_len: int = 64,
    num_runs: int = 20,
) -> Dict[str, Any]:
    """Benchmarks generation forward pass speed: FP32 vs Dynamic INT8."""
    model_cpu = model.cpu()
    model_cpu.eval()
    quantized_model = quantize_model_dynamic(model_cpu)

    dummy_input = torch.randint(0, model.config.vocab_size, (1, seq_len))

    # Profile FP32
    fp32_times: List[float] = []
    with torch.no_grad():
        for _ in range(5):  # warmup
            _ = model_cpu(dummy_input)
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model_cpu(dummy_input)
            fp32_times.append((time.perf_counter() - t0) * 1000.0)

    # Profile INT8
    int8_times: List[float] = []
    with torch.no_grad():
        for _ in range(5):  # warmup
            _ = quantized_model(dummy_input)
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = quantized_model(dummy_input)
            int8_times.append((time.perf_counter() - t0) * 1000.0)

    mean_fp32 = sum(fp32_times) / len(fp32_times)
    mean_int8 = sum(int8_times) / len(int8_times)

    return {
        "fp32_latency_ms": round(mean_fp32, 2),
        "int8_latency_ms": round(mean_int8, 2),
        "speedup_factor": round(mean_fp32 / max(mean_int8, 1e-5), 2),
    }


def benchmark_batch_scaling(
    model: MiniGPT,
    batch_sizes: List[int] = [1, 4, 8, 16, 32],
    seq_len: int = 64,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """Analyzes compute throughput scaling across varying batch dimensions."""
    model.to(device)
    model.eval()
    results: List[Dict[str, Any]] = []

    for B in batch_sizes:
        dummy_input = torch.randint(0, model.config.vocab_size, (B, seq_len), device=device)
        latencies = []
        with torch.no_grad():
            for _ in range(3):  # warmup
                _ = model(dummy_input)
            for _ in range(10):
                t0 = time.perf_counter()
                _ = model(dummy_input)
                latencies.append((time.perf_counter() - t0) * 1000.0)

        mean_lat = sum(latencies) / len(latencies)
        total_tokens = B * seq_len
        tok_sec = (total_tokens / (mean_lat / 1000.0))

        results.append({
            "batch_size": B,
            "latency_ms": round(mean_lat, 2),
            "tokens_per_sec": round(tok_sec, 1),
        })

    return results

