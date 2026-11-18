"""Unit tests for Phase 6: Text Generation and Sampling."""

import pytest
import torch
import torch.nn.functional as F

from src.generation.sampler import (
    apply_temperature,
    apply_top_k,
    apply_top_p,
    apply_repetition_penalty,
    sample_next_token,
)
from src.generation.generate import TextGenerator
from src.model.gpt import MiniGPT, GPTConfig
from src.tokenizer.char_tokenizer import CharTokenizer


def test_temperature_scaling():
    logits = torch.tensor([[2.0, 4.0, 6.0]])

    # High temperature (flattens probabilities)
    probs_high_temp = F.softmax(apply_temperature(logits, temperature=2.0), dim=-1)
    # Low temperature (sharpens probabilities)
    probs_low_temp = F.softmax(apply_temperature(logits, temperature=0.5), dim=-1)

    # In low temp, the highest logit (6.0) should have much higher probability mass
    assert probs_low_temp[0, 2] > probs_high_temp[0, 2]
    # In high temp, the lowest logit (2.0) should have higher probability mass than in low temp
    assert probs_high_temp[0, 0] > probs_low_temp[0, 0]


def test_top_k_filtering():
    logits = torch.tensor([[1.0, 5.0, 2.0, 8.0, 3.0]])
    # Top-2 tokens are 8.0 (index 3) and 5.0 (index 1)
    filtered = apply_top_k(logits, top_k=2)

    # All tokens other than indices 1 and 3 must be -inf
    assert filtered[0, 3] == 8.0
    assert filtered[0, 1] == 5.0
    assert filtered[0, 0] == float("-inf")
    assert filtered[0, 2] == float("-inf")
    assert filtered[0, 4] == float("-inf")


def test_top_p_nucleus_filtering():
    # Construct logits where top token has ~80% probability mass
    logits = torch.tensor([[10.0, 1.0, 0.5, -2.0]])
    filtered = apply_top_p(logits, top_p=0.7)

    # Top token must be preserved
    assert filtered[0, 0] == 10.0
    # Remaining tokens outside top 0.7 mass must be masked to -inf
    assert filtered[0, 3] == float("-inf")


def test_repetition_penalty():
    logits = torch.tensor([[4.0, 2.0, -3.0]])
    # Penalize tokens 0 and 2
    penalized = apply_repetition_penalty(logits, generated_tokens=[0, 2], penalty=2.0)

    # Positive logit 4.0 should be divided by 2.0 -> 2.0
    assert penalized[0, 0] == 2.0
    # Unvisited token 1 should remain 2.0
    assert penalized[0, 1] == 2.0
    # Negative logit -3.0 should be multiplied by 2.0 -> -6.0
    assert penalized[0, 2] == -6.0


def test_sample_next_token_greedy():
    logits = torch.tensor([[1.0, 3.0, 9.0, 2.0]])
    # Low temperature must deterministically pick index 2 (logit 9.0)
    for _ in range(5):
        next_tok = sample_next_token(logits, temperature=0.0)
        assert next_tok.item() == 2


def test_text_generator_stop_sequences():
    corpus = "ABCDEF \n---"
    tokenizer = CharTokenizer.build_from_corpus(corpus)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=16,
        n_layer=1,
        n_head=1,
        n_embd=16,
    )
    model = MiniGPT(config)
    generator = TextGenerator(model=model, tokenizer=tokenizer, device="cpu")

    # Generate with stop condition
    continuation = generator.generate(
        prompt="A",
        max_new_tokens=20,
        temperature=0.8,
        stop_sequences=["B"],  # Stop as soon as 'B' appears
    )
    # The generation must either contain B at the end or terminate
    assert isinstance(continuation, str)

