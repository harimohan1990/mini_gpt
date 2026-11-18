"""Data preparation and normalization pipeline for Mini GPT.

Handles text ingestion, cleaning, unicode normalization, splitting into
train and validation sets without autoregressive leakage, and computing dataset statistics.
"""

from __future__ import annotations

import argparse
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, Tuple


def generate_synthetic_corpus() -> str:
    """Generates an original educational corpus covering dialogues, science,
    and computational reasoning for pre-training Mini GPT.
    """
    topics = [
        # AI and Computing
        (
            "ALICE: What is the core principle behind the Transformer architecture?\n"
            "BOB: The Transformer replaces recurrence with self-attention mechanisms.\n"
            "ALICE: How does self-attention compute relationships between tokens?\n"
            "BOB: For each token, the model projects the embedding into Query, Key, and Value vectors.\n"
            "ALICE: And then it calculates the dot product between Query and Key?\n"
            "BOB: Exactly. The scaled dot product measures affinity, followed by softmax normalization.\n"
            "ALICE: Why is causal masking necessary in decoder-only models?\n"
            "BOB: Causal masking prevents the model from attending to future tokens during training.\n"
            "ALICE: Thus preserving the autoregressive property of next-token prediction.\n"
            "BOB: Precisely. At position t, the network only perceives tokens up to position t.\n"
        ),
        (
            "THEORY: Deep learning models rely on empirical risk minimization.\n"
            "Given a sequence of discrete tokens x_1, x_2, through x_T, the objective is to maximize\n"
            "the log-likelihood of observing the correct continuation.\n"
            "Mathematically, the autoregressive objective decomposes the joint probability:\n"
            "P(x_1, ..., x_T) = product_{t=1}^T P(x_t | x_1, ..., x_{t-1}).\n"
            "Each factor is parameterized by deep neural networks with residual connections\n"
            "and layer normalizations to stabilize gradient propagation across layers.\n"
        ),
        # Astronomy and Physics
        (
            "OBSERVER: Look towards the constellation of Orion in the winter sky.\n"
            "STELLA: The bright red supergiant at the shoulder is Betelgeuse.\n"
            "OBSERVER: It is nearing the terminal phase of stellar nucleosynthesis.\n"
            "STELLA: When core iron fusion halts exothermic release, gravitational collapse triggers a supernova.\n"
            "OBSERVER: The remnant becomes either a neutron star supported by degeneracy pressure, or a black hole.\n"
            "STELLA: Where escape velocity exceeds the speed of light in vacuum.\n"
        ),
        # Philosophy and Logic
        (
            "PHILOSOPHER: Does a formal symbol manipulation system understand semantics?\n"
            "SCHOLAR: Syntax alone cannot induce semantic grounding without embodied or causal interaction.\n"
            "PHILOSOPHER: Yet statistical regularities capture rich world representations.\n"
            "SCHOLAR: Compressibility implies predictability; predictability implies latent modeling of the world.\n"
            "PHILOSOPHER: Therefore, predicting the next word requires simulating the underlying physical reality.\n"
            "SCHOLAR: A miniature universe encoded in continuous geometric vector spaces.\n"
        ),
        # Mathematics and Algorithms
        (
            "ALGORITHM: Binary search achieves logarithmic time complexity O(log N).\n"
            "Given a sorted sequence, maintain low and high pointers.\n"
            "Compute middle index as low plus half the difference between high and low.\n"
            "If target equals middle element, search terminates successfully.\n"
            "If target is strictly smaller, set high to middle minus one.\n"
            "Otherwise set low to middle plus one. Repeat until pointers cross.\n"
        ),
    ]

    # Synthesize a multi-scene training corpus by structured repetition and variations
    corpus_parts = []
    for _ in range(60):
        for scene in topics:
            corpus_parts.append(scene)
            corpus_parts.append("\n---\n\n")

    return "".join(corpus_parts)


def clean_text(raw_text: str, preserve_newlines: bool = True) -> str:
    """Cleans and normalizes raw text.

    Args:
        raw_text: Raw input string.
        preserve_newlines: Whether to retain line breaks for formatting.

    Returns:
        Normalized, cleaned text string.
    """
    # 1. Unicode NFKC normalization (canonical decomposition + canonical composition)
    text = unicodedata.normalize("NFKC", raw_text)

    # 2. Standardize carriage returns and Windows line breaks to standard Unix LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Strip non-printable or dangerous control characters, keeping \n, \t, and standard ASCII/Unicode
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # 4. Normalize multiple horizontal spaces and tabs into single space
    text = re.sub(r"[ \t]+", " ", text)

    # 5. Clean excessive consecutive newlines (keep max 2 newlines for paragraph breaks)
    if preserve_newlines:
        text = re.sub(r"\n{3,}", "\n\n", text)
    else:
        text = re.sub(r"\s+", " ", text)

    return text.strip()


def compute_dataset_stats(text: str) -> Dict[str, int]:
    """Computes basic structural statistics on a text corpus."""
    characters = len(text)
    unique_chars = len(set(text))
    words = len(text.split())
    lines = len(text.splitlines())
    return {
        "characters": characters,
        "unique_characters": unique_chars,
        "words": words,
        "lines": lines,
    }


def split_text(text: str, train_ratio: float = 0.9) -> Tuple[str, str]:
    """Splits a single continuous stream of text into train and val sets.

    Note: Splitting is performed sequentially (chronologically) rather than randomly
    to avoid data leakage across the autoregressive context boundary.
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    split_idx = int(len(text) * train_ratio)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    return train_text, val_text


def prepare_corpus(
    input_path: Path | None = None,
    output_dir: Path = Path("data"),
    train_ratio: float = 0.9,
) -> Tuple[Path, Path, Path]:
    """Prepares raw text, cleans it, splits it, and saves artifacts to disk."""
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_file = raw_dir / "input.txt"
    if input_path is not None and input_path.exists():
        raw_text = input_path.read_text(encoding="utf-8")
    elif raw_file.exists():
        raw_text = raw_file.read_text(encoding="utf-8")
    else:
        raw_text = generate_synthetic_corpus()
        raw_file.write_text(raw_text, encoding="utf-8")

    cleaned_text = clean_text(raw_text)
    train_text, val_text = split_text(cleaned_text, train_ratio=train_ratio)

    train_file = processed_dir / "train.txt"
    val_file = processed_dir / "val.txt"

    train_file.write_text(train_text, encoding="utf-8")
    val_file.write_text(val_text, encoding="utf-8")

    return raw_file, train_file, val_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for Mini GPT")
    parser.add_argument("--input", type=Path, default=None, help="Path to raw text file")
    parser.add_argument("--output_dir", type=Path, default=Path("data"), help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Ratio for train split")
    args = parser.parse_args()

    raw_p, train_p, val_p = prepare_corpus(args.input, args.output_dir, args.train_ratio)
    stats = compute_dataset_stats(train_p.read_text(encoding="utf-8"))
    val_stats = compute_dataset_stats(val_p.read_text(encoding="utf-8"))

    print("=== Mini GPT Dataset Preparation Finished ===")
    print(f"Raw data:       {raw_p}")
    print(f"Train data:     {train_p} -> {stats['characters']:,} chars, {stats['words']:,} words")
    print(f"Val data:       {val_p}   -> {val_stats['characters']:,} chars, {val_stats['words']:,} words")
    print(f"Total Vocab set: {stats['unique_characters']} unique chars in train")

