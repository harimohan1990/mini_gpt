"""Comprehensive evaluation orchestrator for Mini GPT."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import torch

from src.dataset.dataset import get_batch
from src.evaluation.metrics import (
    compute_accuracy,
    compute_bits_per_token,
    compute_distinct_ngrams,
    compute_parameter_breakdown,
)
from src.generation.generate import TextGenerator
from src.model.gpt import MiniGPT
from src.tokenizer.base import BaseTokenizer
from src.training.loss import compute_perplexity, custom_cross_entropy


class ModelEvaluator:
    """Evaluates Mini GPT model across predictive, information-theoretic,
    and generative dimensions.
    """

    def __init__(
        self,
        model: MiniGPT,
        tokenizer: BaseTokenizer,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.generator = TextGenerator(model=model, tokenizer=tokenizer, device=device)

    @torch.no_grad()
    def evaluate_dataset(
        self,
        data: torch.Tensor,
        batch_size: int = 32,
        num_batches: int = 30,
    ) -> Dict[str, float]:
        """Evaluates predictive accuracy, perplexity, and bits-per-token over a dataset."""
        self.model.eval()
        losses: List[float] = []
        top1_accs: List[float] = []
        top3_accs: List[float] = []
        top5_accs: List[float] = []

        block_size = self.model.config.block_size

        for _ in range(num_batches):
            x, y = get_batch(data, block_size=block_size, batch_size=batch_size, device=self.device)
            logits, loss = self.model(x, targets=y)

            losses.append(loss.item())
            top1_accs.append(compute_accuracy(logits, y, top_k=1))
            top3_accs.append(compute_accuracy(logits, y, top_k=3))
            top5_accs.append(compute_accuracy(logits, y, top_k=5))

        mean_loss = sum(losses) / len(losses)
        mean_top1 = sum(top1_accs) / len(top1_accs)
        mean_top3 = sum(top3_accs) / len(top3_accs)
        mean_top5 = sum(top5_accs) / len(top5_accs)

        return {
            "cross_entropy_loss": round(mean_loss, 4),
            "perplexity": round(compute_perplexity(mean_loss), 2),
            "bits_per_token": round(compute_bits_per_token(mean_loss), 3),
            "top1_accuracy": round(mean_top1 * 100.0, 2),
            "top3_accuracy": round(mean_top3 * 100.0, 2),
            "top5_accuracy": round(mean_top5 * 100.0, 2),
        }

    def evaluate_generation_quality(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.8,
    ) -> List[Dict[str, Any]]:
        """Generates continuations and calculates lexical diversity metrics."""
        results: List[Dict[str, Any]] = []

        for p in prompts:
            t0 = time.time()
            completion = self.generator.generate(
                prompt=p,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=20,
                top_p=0.9,
            )
            dt = time.time() - t0

            full_text = p + completion
            d1 = compute_distinct_ngrams(full_text, n=1)
            d2 = compute_distinct_ngrams(full_text, n=2)

            results.append({
                "prompt": p,
                "completion": completion,
                "latency_sec": round(dt, 3),
                "distinct_1": round(d1, 3),
                "distinct_2": round(d2, 3),
            })

        return results

    def run_full_evaluation(
        self,
        val_data: torch.Tensor,
        test_prompts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Executes full evaluation pipeline producing an aggregated report."""
        if test_prompts is None:
            test_prompts = [
                "ALICE: What is the Transformer architecture?",
                "BOB: Deep learning models rely on",
                "THEORY: Causal masking prevents",
            ]

        param_summary = compute_parameter_breakdown(self.model)
        val_metrics = self.evaluate_dataset(val_data)
        gen_metrics = self.evaluate_generation_quality(test_prompts)

        return {
            "parameters": param_summary,
            "validation_metrics": val_metrics,
            "generation_samples": gen_metrics,
        }

