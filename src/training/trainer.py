"""Trainer pipeline for Mini GPT.

Implements:
1. Cosine Learning Rate Schedule with Warmup
2. Gradient Clipping for numerical stability
3. Train and Validation evaluation loops
4. Overfitting detection and tracking
5. Checkpoint saving and resumption
6. Training throughput metrics (tokens/second)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.dataset.dataset import get_batch
from src.training.loss import compute_perplexity

if TYPE_CHECKING:
    from src.model.gpt import MiniGPT


@dataclass
class TrainingConfig:
    """Hyperparameters configuring a training run."""
    max_iters: int = 600
    batch_size: int = 32
    learning_rate: float = 5e-4
    min_lr: float = 5e-5
    warmup_iters: int = 50
    lr_decay_iters: int = 600
    eval_interval: int = 100
    eval_iters: int = 20
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    checkpoint_dir: Path = Path("checkpoints")
    device: str = "auto"
    seed: int = 1337


def get_lr(it: int, config: TrainingConfig) -> float:
    """Calculates learning rate at iteration `it` using cosine schedule with warmup.

    Schedule:
        1. Linear warmup: 0 <= it < warmup_iters:
           lr = max_lr * (it / warmup_iters)
        2. Cosine decay: warmup_iters <= it <= lr_decay_iters:
           decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
           coeff = 0.5 * (1.0 + cos(pi * decay_ratio))
           lr = min_lr + coeff * (max_lr - min_lr)
        3. Post-decay floor: it > lr_decay_iters:
           lr = min_lr
    """
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / (config.warmup_iters + 1)
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


class Trainer:
    """Orchestrates pre-training and evaluation loops for Mini GPT."""

    def __init__(
        self,
        model: MiniGPT,
        optimizer: Optimizer,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        config: TrainingConfig,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_data = train_data
        self.val_data = val_data
        self.config = config

        # Determine optimal device
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = config.device

        self.model.to(self.device)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        self.history: Dict[str, List[float]] = {
            "step": [],
            "train_loss": [],
            "val_loss": [],
            "train_ppl": [],
            "val_ppl": [],
            "lr": [],
        }

    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        """Averages cross-entropy loss and perplexity across multiple batches."""
        self.model.eval()
        out = {}
        splits = [("train", self.train_data), ("val", self.val_data)]

        for split_name, data_source in splits:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                x, y = get_batch(
                    data_source,
                    block_size=self.model.config.block_size,
                    batch_size=self.config.batch_size,
                    device=self.device,
                )
                _, loss = self.model(x, targets=y)
                losses[k] = loss.item()
            mean_loss = losses.mean().item()
            out[f"{split_name}_loss"] = mean_loss
            out[f"{split_name}_ppl"] = compute_perplexity(mean_loss)

        self.model.train()
        return out

    def check_overfitting(self, tolerance_steps: int = 3) -> bool:
        """Detects whether validation loss has consistently risen while train loss fell."""
        if len(self.history["val_loss"]) < tolerance_steps + 1:
            return False

        recent_val = self.history["val_loss"][-tolerance_steps:]
        recent_train = self.history["train_loss"][-tolerance_steps:]

        val_rising = all(recent_val[i] < recent_val[i + 1] for i in range(len(recent_val) - 1))
        train_falling = all(recent_train[i] > recent_train[i + 1] for i in range(len(recent_train) - 1))

        return val_rising and train_falling

    def save_checkpoint(self, filename: str = "best_model.pt", step: int = 0) -> Path:
        """Saves model weights, optimizer state, and metadata to disk."""
        save_path = self.config.checkpoint_dir / filename
        checkpoint = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.model.config,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }
        torch.save(checkpoint, save_path)
        return save_path

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """Loads model weights and optimizer state from a checkpoint file."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.history = checkpoint.get("history", self.history)
        return checkpoint.get("step", 0)

    def train(self) -> Dict[str, List[float]]:
        """Executes the complete training loop."""
        print(f"=== Starting Mini GPT Pre-Training on {self.device.upper()} ===")
        print(f"Model parameters: {self.model.get_num_params():,}")
        print(f"Context window:   {self.model.config.block_size} tokens")
        print(f"Batch size:       {self.config.batch_size} (Tokens/step: {self.config.batch_size * self.model.config.block_size:,})")
        print(f"Total iterations: {self.config.max_iters}\n")

        self.model.train()
        start_time = time.time()
        tokens_processed = 0

        for it in range(self.config.max_iters):
            t0 = time.time()

            # 1. Update learning rate according to cosine schedule
            lr = get_lr(it, self.config)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # 2. Sample dynamic batch
            x, y = get_batch(
                self.train_data,
                block_size=self.model.config.block_size,
                batch_size=self.config.batch_size,
                device=self.device,
            )

            # 3. Forward pass & loss computation
            logits, loss = self.model(x, targets=y)

            # 4. Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # 5. Gradient clipping to prevent exploding gradients
            if self.config.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            # 6. Optimizer step
            self.optimizer.step()

            # Track tokens processed
            step_tokens = self.config.batch_size * self.model.config.block_size
            tokens_processed += step_tokens
            dt = time.time() - t0

            # Periodic evaluation
            if it % self.config.eval_interval == 0 or it == self.config.max_iters - 1:
                metrics = self.estimate_loss()
                train_loss, val_loss = metrics["train_loss"], metrics["val_loss"]
                train_ppl, val_ppl = metrics["train_ppl"], metrics["val_ppl"]

                self.history["step"].append(it)
                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["train_ppl"].append(train_ppl)
                self.history["val_ppl"].append(val_ppl)
                self.history["lr"].append(lr)

                tok_per_sec = step_tokens / max(dt, 1e-5)
                print(
                    f"Step {it:4d}/{self.config.max_iters} | "
                    f"Train Loss: {train_loss:.4f} (PPL: {train_ppl:5.1f}) | "
                    f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:5.1f}) | "
                    f"LR: {lr:.2e} | "
                    f"{tok_per_sec:,.0f} tok/s"
                )

                # Check for best validation score and checkpoint
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_path = self.save_checkpoint("best_model.pt", step=it)
                    print(f"  --> Saved new best checkpoint to {best_path} (Val Loss: {val_loss:.4f})")

                # Detect overfitting warning
                if self.check_overfitting():
                    print("  [!] WARNING: Potential overfitting detected (validation loss diverging from train loss)!")

        total_time = time.time() - start_time
        latest_path = self.save_checkpoint("latest_model.pt", step=self.config.max_iters)
        print(f"\n=== Training Complete in {total_time:.2f}s ===")
        print(f"Total tokens processed: {tokens_processed:,}")
        print(f"Best Validation Loss:  {self.best_val_loss:.4f}")
        print(f"Final checkpoint saved: {latest_path}")

        return self.history
