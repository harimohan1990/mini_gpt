"""Model Quantization utilities for Mini GPT.

Implements:
1. Dynamic Post-Training Quantization (INT8)
2. Mathematical scale and zero-point quantization formulation
3. Model size and memory footprint comparisons
4. Serialization of quantized models
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple
import torch
import torch.nn as nn

from src.model.gpt import MiniGPT


def quantize_model_dynamic(model: MiniGPT) -> nn.Module:
    """Applies dynamic INT8 post-training quantization to Linear layers."""
    # Ensure a valid quantization engine is selected (e.g. qnnpack for ARM/x86)
    if "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"

    model_cpu = model.cpu()
    model_cpu.eval()

    # Dynamic quantization converts nn.Linear layers to torch.qint8
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return quantized_model


def get_model_file_size(model: nn.Module) -> float:
    """Measures the on-disk serialization size of a model in Megabytes (MB)."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            torch.save(model.state_dict(), tmp_path)
            size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    return round(size_mb, 3)


def compare_quantization(model: MiniGPT) -> Dict[str, float]:
    """Compares size and memory reduction between FP32 baseline and INT8 quantized model."""
    fp32_size = get_model_file_size(model)
    quantized_model = quantize_model_dynamic(model)
    int8_size = get_model_file_size(quantized_model)

    compression_ratio = round(fp32_size / max(int8_size, 1e-5), 2)
    memory_savings_pct = round((1.0 - (int8_size / max(fp32_size, 1e-5))) * 100.0, 1)

    return {
        "fp32_size_mb": fp32_size,
        "int8_size_mb": int8_size,
        "compression_ratio": compression_ratio,
        "memory_savings_pct": memory_savings_pct,
    }
