"""Unit tests for Phase 5: Model Training and Optimization."""

import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.model.gpt import MiniGPT, GPTConfig
from src.training.optimizer import configure_optimizers
from src.training.trainer import Trainer, TrainingConfig, get_lr


def test_get_lr_schedule():
    config = TrainingConfig(
        max_iters=1000,
        learning_rate=1e-3,
        min_lr=1e-4,
        warmup_iters=100,
        lr_decay_iters=1000,
    )

    # 1. Warmup checks
    lr_0 = get_lr(0, config)
    lr_50 = get_lr(50, config)
    lr_100 = get_lr(100, config)

    assert lr_0 < lr_50 < lr_100
    assert pytest.approx(lr_100, rel=1e-2) == 1e-3

    # 2. Cosine decay checks
    lr_500 = get_lr(500, config)
    assert 1e-4 < lr_500 < 1e-3

    # 3. Post-decay floor
    lr_1050 = get_lr(1050, config)
    assert lr_1050 == 1e-4


def test_gradient_clipping():
    model = nn.Linear(10, 10)
    x = torch.randn(5, 10)
    # Produce massive loss to trigger huge gradients
    loss = (model(x) * 1e6).sum()
    loss.backward()

    initial_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    assert initial_norm > 1.0

    # Total norm after clipping must be <= 1.0 + epsilon
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters()]), 2)
    assert total_norm <= 1.0001


def test_overfit_single_batch():
    """Sanity check: A healthy GPT model must easily overfit a single tiny batch."""
    torch.manual_seed(42)
    config = GPTConfig(
        vocab_size=20,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        tie_weights=True,
    )
    model = MiniGPT(config)
    optimizer = configure_optimizers(model, weight_decay=0.0, learning_rate=3e-3)

    x = torch.randint(0, 20, (2, 8))
    y = torch.randint(0, 20, (2, 8))

    initial_loss = float("inf")
    final_loss = float("inf")

    model.train()
    for step in range(35):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y)
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    # The loss must decrease significantly
    assert final_loss < initial_loss * 0.2
    assert final_loss < 0.5


def test_checkpoint_save_and_load(tmp_path: Path):
    torch.manual_seed(42)
    config = GPTConfig(
        vocab_size=25,
        block_size=8,
        n_layer=1,
        n_head=1,
        n_embd=16,
    )
    model = MiniGPT(config)
    optimizer = configure_optimizers(model, learning_rate=1e-3)

    dummy_data = torch.randint(0, 25, (100,))
    train_config = TrainingConfig(
        max_iters=5,
        batch_size=2,
        checkpoint_dir=tmp_path,
        device="cpu",
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_data=dummy_data,
        val_data=dummy_data,
        config=train_config,
    )

    ckpt_path = trainer.save_checkpoint("test_ckpt.pt", step=42)
    assert ckpt_path.exists()

    # Fresh model
    model_fresh = MiniGPT(config)
    optimizer_fresh = configure_optimizers(model_fresh, learning_rate=1e-3)
    trainer_fresh = Trainer(
        model=model_fresh,
        optimizer=optimizer_fresh,
        train_data=dummy_data,
        val_data=dummy_data,
        config=train_config,
    )

    step_loaded = trainer_fresh.load_checkpoint(ckpt_path)
    assert step_loaded == 42

    # Forward pass outputs must be identical in eval mode (disabling dropout)
    model.eval()
    model_fresh.eval()
    test_input = torch.randint(0, 25, (1, 8))
    with torch.no_grad():
        logits_orig, _ = model(test_input)
        logits_loaded, _ = model_fresh(test_input)

    assert torch.allclose(logits_orig, logits_loaded, atol=1e-6)
