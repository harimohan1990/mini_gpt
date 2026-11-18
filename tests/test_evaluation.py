"""Unit tests for Phase 8: Model Evaluation and Metrics."""

import math
import pytest
import torch
import torch.nn as nn

from src.evaluation.metrics import (
    compute_accuracy,
    compute_bits_per_token,
    compute_distinct_ngrams,
    compute_parameter_breakdown,
)
from src.evaluation.evaluator import ModelEvaluator
from src.model.gpt import MiniGPT, GPTConfig
from src.tokenizer.char_tokenizer import CharTokenizer


def test_compute_accuracy_top1_and_topk():
    # Construct 4 samples over 3 classes:
    # Sample 0: true = 0, logits = [10, 0, 0] -> correct
    # Sample 1: true = 1, logits = [0, 10, 0] -> correct
    # Sample 2: true = 2, logits = [0, 10, 5] -> incorrect for top-1 (picks 1), correct for top-2
    # Sample 3: true = 1, logits = [5, 0, 10] -> incorrect for top-1 and top-2
    logits = torch.tensor([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 10.0, 5.0],
        [5.0, 0.0, 10.0],
    ])
    targets = torch.tensor([0, 1, 2, 1])

    acc_1 = compute_accuracy(logits, targets, top_k=1)
    # Correct top-1: 0, 1 (2 out of 4 = 50.0%)
    assert pytest.approx(acc_1, rel=1e-3) == 0.50

    acc_2 = compute_accuracy(logits, targets, top_k=2)
    # Correct top-2: 0, 1, 2 (3 out of 4 = 75.0%)
    assert pytest.approx(acc_2, rel=1e-3) == 0.75


def test_compute_accuracy_with_ignore_index():
    logits = torch.tensor([
        [10.0, 0.0],
        [0.0, 10.0],
    ])
    # Target 1 is ignored
    targets = torch.tensor([0, -100])
    acc = compute_accuracy(logits, targets, top_k=1, ignore_index=-100)
    # Only 1 valid sample, which was correct -> 100%
    assert acc == 1.0


def test_compute_bits_per_token():
    # If cross entropy loss is ln(2), bits per token should be exactly 1.0
    loss = math.log(2.0)
    bpc = compute_bits_per_token(loss)
    assert pytest.approx(bpc, rel=1e-5) == 1.0

    loss_4 = math.log(16.0)  # log2(16) = 4 bits
    assert pytest.approx(compute_bits_per_token(loss_4), rel=1e-5) == 4.0


def test_compute_distinct_ngrams():
    # Repetitive text: "a a a a" -> 1 unique unigram out of 4 tokens -> 0.25
    text_rep = "a a a a"
    assert pytest.approx(compute_distinct_ngrams(text_rep, n=1), rel=1e-3) == 0.25

    # Unique text: "a b c d" -> 4 unique unigrams out of 4 tokens -> 1.0
    text_uniq = "a b c d"
    assert pytest.approx(compute_distinct_ngrams(text_uniq, n=1), rel=1e-3) == 1.0


def test_compute_parameter_breakdown():
    config = GPTConfig(
        vocab_size=30,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )
    model = MiniGPT(config)
    breakdown = compute_parameter_breakdown(model)

    assert breakdown["total_parameters"] > 0
    assert breakdown["trainable_parameters"] == breakdown["total_parameters"]
    assert "embeddings" in breakdown["categories"]
    assert "attention" in breakdown["categories"]
    assert "feedforward" in breakdown["categories"]
    assert breakdown["memory_fp32_mb"] > 0.0


def test_model_evaluator_end_to_end():
    corpus = "ABCDEF 01234\n"
    tokenizer = CharTokenizer.build_from_corpus(corpus)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=8,
        n_layer=1,
        n_head=1,
        n_embd=16,
    )
    model = MiniGPT(config)
    dummy_data = torch.randint(0, tokenizer.vocab_size, (100,))

    evaluator = ModelEvaluator(model=model, tokenizer=tokenizer, device="cpu")
    metrics = evaluator.evaluate_dataset(dummy_data, batch_size=2, num_batches=3)

    assert "cross_entropy_loss" in metrics
    assert "perplexity" in metrics
    assert "top1_accuracy" in metrics
    assert "bits_per_token" in metrics
    assert 0.0 <= metrics["top1_accuracy"] <= 100.0

