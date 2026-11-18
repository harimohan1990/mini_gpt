"""Top-level CLI evaluation script for Mini GPT."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch

from src.evaluation.evaluator import ModelEvaluator
from src.model.gpt import MiniGPT
from src.tokenizer.char_tokenizer import CharTokenizer
from src.tokenizer.bpe_tokenizer import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Mini GPT Checkpoint")
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
        help="Path to tokenizer JSON file",
    )
    parser.add_argument(
        "--val_data_path",
        type=Path,
        default=Path("data/processed/val.txt"),
        help="Path to validation text file",
    )
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cpu', 'cuda', 'mps')")
    parser.add_argument("--num_batches", type=int, default=30, help="Number of validation batches to evaluate")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}. Please train model first.")
    if not args.val_data_path.exists():
        raise FileNotFoundError(f"Validation data not found at {args.val_data_path}.")

    # 1. Device resolution
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # 2. Load tokenizer
    if "bpe" in str(args.tokenizer_path).lower():
        tokenizer = BPETokenizer.load(args.tokenizer_path)
    else:
        tokenizer = CharTokenizer.load(args.tokenizer_path)

    # 3. Load model checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = MiniGPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 4. Ingest and tokenize validation data
    val_text = args.val_data_path.read_text(encoding="utf-8")
    val_data = tokenizer.encode_to_tensor(val_text)

    # 5. Run evaluation
    evaluator = ModelEvaluator(model=model, tokenizer=tokenizer, device=device)
    report = evaluator.run_full_evaluation(val_data=val_data)

    # 6. Display formatted evaluation report
    params = report["parameters"]
    v_metrics = report["validation_metrics"]
    gen_samples = report["generation_samples"]

    print("\n" + "=" * 65)
    print("           MINI GPT RESEARCH EVALUATION REPORT")
    print("=" * 65)

    print("\n[1] MODEL ARCHITECTURE & MEMORY INVENTORY")
    print("-" * 50)
    print(f"Total Parameters:       {params['total_parameters']:,}")
    print(f"Trainable Parameters:   {params['trainable_parameters']:,}")
    print(f"Embedding Parameters:   {params['categories'].get('embeddings', 0):,}")
    print(f"Attention Parameters:   {params['categories'].get('attention', 0):,}")
    print(f"MLP (FFN) Parameters:   {params['categories'].get('feedforward', 0):,}")
    print(f"LayerNorm Parameters:   {params['categories'].get('layernorm', 0):,}")
    print(f"Memory Footprint (FP32):{params['memory_fp32_mb']} MB")
    print(f"Memory Footprint (FP16):{params['memory_fp16_mb']} MB")
    print(f"Memory Footprint (INT8):{params['memory_int8_mb']} MB")

    print("\n[2] VALIDATION SET PREDICTIVE PERFORMANCE")
    print("-" * 50)
    print(f"Cross-Entropy Loss:     {v_metrics['cross_entropy_loss']:.4f} nats")
    print(f"Perplexity (PPL):       {v_metrics['perplexity']:.2f}")
    print(f"Bits-Per-Token (BPC):   {v_metrics['bits_per_token']:.3f} bits")
    print(f"Top-1 Token Accuracy:   {v_metrics['top1_accuracy']:.2f}%")
    print(f"Top-3 Token Accuracy:   {v_metrics['top3_accuracy']:.2f}%")
    print(f"Top-5 Token Accuracy:   {v_metrics['top5_accuracy']:.2f}%")

    print("\n[3] GENERATION QUALITY & LEXICAL DIVERSITY")
    print("-" * 50)
    for i, s in enumerate(gen_samples, 1):
        print(f"\nSample #{i}:")
        print(f"  Prompt:      \"{s['prompt']}\"")
        print(f"  Continuation:\"{s['completion'][:100]}...\"")
        print(f"  Distinct-1:  {s['distinct_1']} | Distinct-2: {s['distinct_2']} | Latency: {s['latency_sec']}s")

    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    main()

