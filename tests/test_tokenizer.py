"""Unit tests for Phase 2: Tokenization (Character and BPE)."""

import json
from pathlib import Path
import pytest
import torch

from src.tokenizer.char_tokenizer import CharTokenizer
from src.tokenizer.bpe_tokenizer import BPETokenizer


def test_char_tokenizer_build_and_vocab():
    corpus = "hello world"
    tok = CharTokenizer.build_from_corpus(corpus)

    # Unique characters: ' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w' (8 chars)
    # Special tokens: '<unk>', '<pad>', '<bos>', '<eos>' (4 tokens)
    # Total vocab size = 12
    assert tok.vocab_size == 12
    assert len(tok.token_to_id) == 12
    assert len(tok.id_to_token) == 12

    # Bijective mapping assertion
    for token, idx in tok.token_to_id.items():
        assert tok.id_to_token[idx] == token


def test_char_tokenizer_encode_decode_roundtrip():
    corpus = "abcdefghijklmnopqrstuvwxyz 0123456789\n"
    tok = CharTokenizer.build_from_corpus(corpus)

    original_text = "hello world\n"
    encoded = tok.encode(original_text)
    decoded = tok.decode(encoded)

    assert decoded == original_text
    assert isinstance(encoded, list)
    assert all(isinstance(x, int) for x in encoded)


def test_char_tokenizer_unknown_token():
    corpus = "abc"
    tok = CharTokenizer.build_from_corpus(corpus, unk_token="<unk>")

    # 'z' is not in vocabulary, should map to <unk>
    encoded = tok.encode("abz")
    assert encoded[-1] == tok.unk_id

    # Decoding with skip_special_tokens
    decoded_skip = tok.decode(encoded, skip_special_tokens=True)
    assert decoded_skip == "ab"


def test_char_tokenizer_tensor_methods():
    corpus = "Transformer"
    tok = CharTokenizer.build_from_corpus(corpus)

    tensor = tok.encode_to_tensor("Trans", device="cpu")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.long
    assert tensor.ndim == 1
    assert len(tensor) == 5

    reconstructed = tok.decode_from_tensor(tensor)
    assert reconstructed == "Trans"


def test_char_tokenizer_save_and_load(tmp_path: Path):
    corpus = "Save and Load Mini GPT"
    tok = CharTokenizer.build_from_corpus(corpus)
    save_path = tmp_path / "tokenizer.json"

    tok.save(save_path)
    assert save_path.exists()

    loaded_tok = CharTokenizer.load(save_path)
    assert loaded_tok.vocab_size == tok.vocab_size
    assert loaded_tok.token_to_id == tok.token_to_id

    test_str = "Mini GPT"
    assert loaded_tok.encode(test_str) == tok.encode(test_str)
    assert loaded_tok.decode(tok.encode(test_str)) == test_str


def test_bpe_tokenizer_training_and_merges():
    # Repetitive patterns where 'th' and 'the' should merge first
    corpus = "the theory of the theme is the thesis of the theory"
    target_vocab_size = 20  # Base chars + specials + merges

    bpe = BPETokenizer.train(corpus, vocab_size=target_vocab_size)
    assert bpe.vocab_size == target_vocab_size
    assert len(bpe.merges) > 0

    # Ensure 'th' or 'the' was learned as a merged subword
    assert any("th" in merge or "the" in merge for merge in bpe.merges or bpe.token_to_id)

    encoded = bpe.encode("the theory")
    decoded = bpe.decode(encoded)
    assert decoded == "the theory"


def test_bpe_tokenizer_save_and_load(tmp_path: Path):
    corpus = "apple banana apple orange banana apple"
    bpe = BPETokenizer.train(corpus, vocab_size=18)
    save_file = tmp_path / "bpe.json"

    bpe.save(save_file)
    assert save_file.exists()

    loaded_bpe = BPETokenizer.load(save_file)
    assert loaded_bpe.vocab_size == bpe.vocab_size
    assert loaded_bpe.merges == bpe.merges

    text = "apple banana"
    assert loaded_bpe.encode(text) == bpe.encode(text)
    assert loaded_bpe.decode(bpe.encode(text)) == text

