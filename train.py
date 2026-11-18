"""Top-level training CLI script for Mini GPT."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch

from src.tokenizer.char_tokenizer import CharTokenizer
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.model.gpt import MiniGPT, GPTConfig
from src.training.optimizer import configure_optimizers
from src.training.trainer import Trainer, TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Mini GPT from scratch")
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed"), help="Path to processed data directory")
    parser.add_argument("--tokenizer_type", type=str, default="char", choices=["char", "bpe"], help="Tokenizer type")
    parser.add_argument("--n_layer", type=int, default=4, help="Number of Transformer blocks")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=128, help="Embedding / hidden dimension")
    parser.add_argument("--block_size", type=int, default=128, help="Context length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_iters", type=int, default=500, help="Total training steps")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cpu', 'cuda', 'mps')")
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"), help="Checkpoint directory")
    args = parser.parse_args()

    # 1. Load tokenizer
    tok_file = args.data_dir / f"{args.tokenizer_type}_tokenizer.json"
    if not tok_file.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tok_file}. Run tokenizer training first.")

    if args.tokenizer_type == "char":
        tokenizer = CharTokenizer.load(tok_file)
    else:
        tokenizer = BPETokenizer.load(tok_file)

    print(f"Loaded {args.tokenizer_type.upper()} tokenizer with vocabulary size: {tokenizer.vocab_size}")

    # 2. Ingest and tokenize train & val datasets
    train_txt_path = args.data_dir / "train.txt"
    val_txt_path = args.data_dir / "val.txt"
    if not train_txt_path.exists() or not val_txt_path.exists():
        raise FileNotFoundError(f"Data files missing in {args.data_dir}. Run dataset preparation first.")

    train_text = train_txt_path.read_text(encoding="utf-8")
    val_text = val_txt_path.read_text(encoding="utf-8")

    train_data = tokenizer.encode_to_tensor(train_text)
    val_data = tokenizer.encode_to_tensor(val_text)
    print(f"Dataset tokenized: {len(train_data):,} train tokens, {len(val_data):,} val tokens")

    # 3. Instantiate model
    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.1,
        tie_weights=True,
    )
    model = MiniGPT(model_config)

    # 4. Configure optimizer with decoupled weight decay
    optimizer = configure_optimizers(
        model,
        weight_decay=0.1,
        learning_rate=args.lr,
        betas=(0.9, 0.95),
    )

    # 5. Training configuration
    train_config = TrainingConfig(
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        min_lr=args.lr * 0.1,
        warmup_iters=max(20, args.max_iters // 10),
        lr_decay_iters=args.max_iters,
        eval_interval=max(50, args.max_iters // 5),
        eval_iters=15,
        grad_clip=1.0,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        val_data=val_data,
        config=train_config,
    )

    # 6. Run training loop
    trainer.train()

    # 7. Generate a qualitative sample
    print("\n--- Generating Sample from Trained Model ---")
    prompt_str = "ALICE: What is" if args.tokenizer_type == "char" else "What is"
    prompt_ids = tokenizer.encode_to_tensor(prompt_str, device=trainer.device).unsqueeze(0)
    generated_ids = model.generate(prompt_ids, max_new_tokens=150, temperature=0.8, top_k=10)
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(generated_text)
    print("-------------------------------------------\n")


if __name__ == "__main__":
    main()

