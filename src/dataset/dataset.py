"""PyTorch Dataset, Dataloader, and Batching for Mini GPT.

Explains and implements:
1. Context window / Sequence length (block_size, T)
2. Batch size (B)
3. Input sequence (X) and Target sequence (Y) with 1-step right-shift
4. PyTorch Dataset & DataLoader
5. Fast random-chunk batch sampler (get_batch)
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple


class TextDataset(Dataset):
    """PyTorch Dataset for autoregressive language modeling.

    Given a 1D tensor of token IDs [s_0, s_1, ..., s_N], this dataset yields:
        x = [s_i,     s_{i+1}, ..., s_{i+T-1}]  (Input context)
        y = [s_{i+1}, s_{i+2}, ..., s_{i+T}]    (Target next tokens)

    where T is the block_size (context window).
    """

    def __init__(self, data: torch.Tensor, block_size: int) -> None:
        """Args:
            data: 1D torch.Tensor of shape (N,) containing encoded token IDs.
            block_size: Maximum sequence length / context window (T).
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"data must be a torch.Tensor, got {type(data)}")
        if data.ndim != 1:
            raise ValueError(f"data must be 1-dimensional, got shape {data.shape}")
        if len(data) <= block_size:
            raise ValueError(
                f"Data length ({len(data)}) must be strictly greater than block_size ({block_size})"
            )

        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        """Returns the total number of valid consecutive sequences of length block_size."""
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves a single (input, target) sequence pair.

        Returns:
            x: Input tensor of shape (block_size,) of dtype torch.long
            y: Target tensor of shape (block_size,) of dtype torch.long (shifted by 1)
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self)}")

        # Input: tokens from idx to idx + block_size
        x = self.data[idx : idx + self.block_size].long()
        # Target: next tokens from idx + 1 to idx + block_size + 1
        y = self.data[idx + 1 : idx + self.block_size + 1].long()
        return x, y


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates a small batch of inputs X and targets Y using random offsets.

    This is the idiomatic, memory-efficient way to sample batches during
    GPT pretraining without materializing billions of overlapping windows.

    Mathematical Formulation:
        Let N = len(data), T = block_size, B = batch_size.
        Sample B random offsets:
            i_b ~ Uniform(0, N - T - 1) for b in 0..B-1
        Construct:
            X[b, t] = data[i_b + t]
            Y[b, t] = data[i_b + t + 1]

    Args:
        data: 1D torch.Tensor of encoded token IDs of shape (N,).
        block_size: Sequence length T (context window).
        batch_size: Number of independent sequences per batch B.
        device: Device to place the tensors on ('cpu', 'cuda', 'mps').

    Returns:
        x: Tensor of shape (B, T) representing input token IDs.
        y: Tensor of shape (B, T) representing target next-token IDs.
    """
    if len(data) <= block_size:
        raise ValueError(f"Data length ({len(data)}) must be > block_size ({block_size})")

    # Generate random starting indices uniformly from [0, len(data) - block_size - 1]
    max_idx = len(data) - block_size
    ix = torch.randint(low=0, high=max_idx, size=(batch_size,))

    # Stack chunks into batch tensors of shape (B, T)
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device=device, dtype=torch.long)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device=device, dtype=torch.long)

    return x, y


def create_dataloader(
    dataset: TextDataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Creates a standard PyTorch DataLoader wrapping a TextDataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

