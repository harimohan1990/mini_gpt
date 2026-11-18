"""Top-level CLI benchmarking script exploring Phase 9 optimizations."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch

from src.model.gpt import MiniGPT, GPTConfig
from src.optimization.benchmark import (
    benchmark_attention_efficiency,
    benchmark_batch_scaling,
    benchmark_device_inference,
    benchmark_quantization_latency,
)
from src.optimization.quantization import compare_quantization


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Mini GPT Optimizations")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_model.pt"),
        help="Path to trained model checkpoint",
    )
    args = parser.parse_args()

    # 1. Device detection
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")

    print("\n" + "=" * 65)
    print("           MINI GPT RESEARCH OPTIMIZATION BENCHMARK")
    print("=" * 65)
    print(f"Available Devices Detected: {', '.join([d.upper() for d in devices])}\n")

    # 2. Load model
    if args.checkpoint.exists():
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = MiniGPT(ckpt["config"])
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        config = GPTConfig(vocab_size=70, block_size=64, n_layer=4, n_head=4, n_embd=128)
        model = MiniGPT(config)

    # -------------------------------------------------------------
    # 1. Attention Engine: Manual Attention vs PyTorch SDPA
    # -------------------------------------------------------------
    print("[1] EFFICIENT ATTENTION PROFILING (FlashAttention / SDPA)")
    print("-" * 55)
    attn_res = benchmark_attention_efficiency(seq_len=128, batch_size=16)
    print(f"Standard Manual Attention Latency: {attn_res['manual_attention_ms']:.3f} ms")
    print(f"PyTorch SDPA Kernel Latency:       {attn_res['sdpa_attention_ms']:.3f} ms")
    print(f"Attention Compute Speedup:          {attn_res['speedup_factor']}x")

    # -------------------------------------------------------------
    # 2. Dynamic INT8 Quantization
    # -------------------------------------------------------------
    print("\n[2] DYNAMIC POST-TRAINING QUANTIZATION (INT8 vs FP32)")
    print("-" * 55)
    quant_size = compare_quantization(model)
    quant_perf = benchmark_quantization_latency(model, seq_len=64)
    print(f"Baseline FP32 Model Size:          {quant_size['fp32_size_mb']} MB")
    print(f"Quantized INT8 Model Size:         {quant_size['int8_size_mb']} MB")
    print(f"Memory Compression Ratio:          {quant_size['compression_ratio']}x ({quant_size['memory_savings_pct']}% reduction)")
    print(f"FP32 Forward Pass Latency:         {quant_perf['fp32_latency_ms']} ms")
    print(f"INT8 Forward Pass Latency:         {quant_perf['int8_latency_ms']} ms")

    # -------------------------------------------------------------
    # 3. Batch Size Scaling & Throughput
    # -------------------------------------------------------------
    print("\n[3] BATCH SIZE SCALING & THROUGHPUT ANALYSIS (CPU)")
    print("-" * 55)
    print("Batch Size | Latency (ms) | Throughput (tokens/sec)")
    print("-" * 55)
    batch_res = benchmark_batch_scaling(model, batch_sizes=[1, 4, 8, 16, 32], seq_len=64)
    for b in batch_res:
        print(f"  {b['batch_size']:4d}     |   {b['latency_ms']:7.2f}    |  {b['tokens_per_sec']:10,.1f}")

    # -------------------------------------------------------------
    # 4. Multi-Device Latency Benchmarks
    # -------------------------------------------------------------
    print("\n[4] HARDWARE DEVICE LATENCY COMPARISON")
    print("-" * 55)
    for dev in devices:
        dev_res = benchmark_device_inference(model, device=dev, seq_len=64)
        print(f"Device: {dev.upper():5s} | Mean: {dev_res['mean_latency_ms']:6.2f} ms | Throughput: {dev_res['throughput_tokens_per_sec']:,.1f} tok/s")

    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    main()

