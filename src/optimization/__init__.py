"""Optimization and acceleration utilities for Mini GPT."""

from src.optimization.quantization import (
    quantize_model_dynamic,
    get_model_file_size,
    compare_quantization,
)
from src.optimization.benchmark import (
    benchmark_device_inference,
    benchmark_attention_efficiency,
    benchmark_quantization_latency,
    benchmark_batch_scaling,
)

__all__ = [
    "quantize_model_dynamic",
    "get_model_file_size",
    "compare_quantization",
    "benchmark_device_inference",
    "benchmark_attention_efficiency",
    "benchmark_quantization_latency",
    "benchmark_batch_scaling",
]

