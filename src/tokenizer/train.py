"""CLI script to build and save tokenizers for Mini GPT."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tokenizer.char_tokenizer import CharTokenizer
from src.tokenizer.bpe_tokenizer import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Mini GPT Tokenizers")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("data/processed/train.txt"),
        help="Path to training text corpus",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to save tokenizer artifacts",
    )
    parser.add_argument(
        "--bpe_vocab_size",
        type=int,
        default=256,
        help="Target vocabulary size for BPE tokenizer",
    )
    args = parser.parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(f"Training data not found at {args.data_path}. Run dataset preparation first.")

    text = args.data_path.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build and save Character Tokenizer
    char_tok = CharTokenizer.build_from_corpus(text)
    char_save_path = args.output_dir / "char_tokenizer.json"
    char_tok.save(char_save_path)
    print(f"Character Tokenizer saved to: {char_save_path}")
    print(f"  Vocabulary size: {char_tok.vocab_size} tokens")

    # 2. Train and save BPE Tokenizer
    bpe_tok = BPETokenizer.train(text, vocab_size=args.bpe_vocab_size)
    bpe_save_path = args.output_dir / "bpe_tokenizer.json"
    bpe_tok.save(bpe_save_path)
    print(f"BPE Tokenizer saved to: {bpe_save_path}")
    print(f"  Vocabulary size: {bpe_tok.vocab_size} tokens, Merges: {len(bpe_tok.merges)}")


if __name__ == "__main__":
    main()

