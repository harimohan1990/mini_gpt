"""Top-level CLI text generation script for Mini GPT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.tokenizer.char_tokenizer import CharTokenizer
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.generation.generate import TextGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text using trained Mini GPT")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_model.pt"),
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=Path,
        default=Path("data/processed/char_tokenizer.json"),
        help="Path to tokenizer JSON artifact",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="ALICE: What is the Transformer architecture?",
        help="Input text prompt to condition generation",
    )
    parser.add_argument("--max_tokens", type=int, default=150, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=20, help="Top-K filtering cutoff")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P (Nucleus) cumulative mass cutoff")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty (>1.0)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cpu', 'cuda', 'mps')")
    parser.add_argument("--stream", action="store_true", default=True, help="Stream tokens to stdout in real time")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {args.checkpoint}. Please train the model first with 'python3 train.py'."
        )

    # 1. Load tokenizer
    if "bpe" in str(args.tokenizer_path).lower():
        tokenizer = BPETokenizer.load(args.tokenizer_path)
    else:
        tokenizer = CharTokenizer.load(args.tokenizer_path)

    # 2. Instantiate TextGenerator from checkpoint
    generator = TextGenerator.from_checkpoint(
        checkpoint_path=args.checkpoint,
        tokenizer=tokenizer,
        device=args.device,
    )

    print("\n================ Mini GPT Generation ================")
    print(f"Device:             {generator.device.upper()}")
    print(f"Sampling:           Temp={args.temperature}, Top-K={args.top_k}, Top-P={args.top_p}, RepPenalty={args.repetition_penalty}")
    print(f"Prompt:             {args.prompt}")
    print("-------------------- Continuation --------------------\n")

    sys.stdout.write(args.prompt)
    sys.stdout.flush()

    if args.stream:
        for token_chunk in generator.stream_generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop_sequences=["\n\n\n"],
        ):
            sys.stdout.write(token_chunk)
            sys.stdout.flush()
    else:
        continuation = generator.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        sys.stdout.write(continuation)

    print("\n\n======================================================\n")


if __name__ == "__main__":
    main()

