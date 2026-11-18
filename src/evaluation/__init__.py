"""Evaluation metrics and pipelines for Mini GPT."""

from src.evaluation.metrics import (
    compute_accuracy,
    compute_bits_per_token,
    compute_distinct_ngrams,
    compute_parameter_breakdown,
)
from src.evaluation.evaluator import ModelEvaluator

__all__ = [
    "compute_accuracy",
    "compute_bits_per_token",
    "compute_distinct_ngrams",
    "compute_parameter_breakdown",
    "ModelEvaluator",
]

