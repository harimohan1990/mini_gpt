"""Unit tests for Phase 1: Dataset Preparation and Batching."""

import pytest
import torch
from pathlib import Path

from src.dataset.prepare import clean_text, split_text, compute_dataset_stats, prepare_corpus
from src.dataset.dataset import TextDataset, get_batch, create_dataloader


def test_clean_text_normalization():
    # Test NFKC normalization (e.g. ligature ﬁ -> fi)
    raw = "The ﬁnal \r\nfrontier\twith   extra   spaces\r\n\r\n\r\n\r\nEnd."
    cleaned = clean_text(raw, preserve_newlines=True)

    assert "final" in cleaned
    assert "\r" not in cleaned
    assert "extra spaces" in cleaned
    assert "\n\n\n" not in cleaned


def test_clean_text_control_characters():
    # Embed non-printable ASCII control characters like bell (\x07) and backspace (\x08)
    raw = "Valid text\x07 with hidden\x08 control codes."
    cleaned = clean_text(raw)
    assert "\x07" not in cleaned
    assert "\x08" not in cleaned
    assert cleaned == "Valid text with hidden control codes."


def test_split_text():
    text = "0123456789"
    train_text, val_text = split_text(text, train_ratio=0.8)
    assert train_text == "01234567"
    assert val_text == "89"
    assert len(train_text) + len(val_text) == len(text)

    # Invalid ratio test
    with pytest.raises(ValueError):
        split_text(text, train_ratio=1.5)


def test_compute_dataset_stats():
    text = "Hello world\nThis is Mini GPT."
    stats = compute_dataset_stats(text)
    assert stats["lines"] == 2
    assert stats["words"] == 6
    assert stats["characters"] == len(text)
    assert stats["unique_characters"] == len(set(text))


def test_text_dataset_shapes_and_alignment():
    # Sequence of integers 0 to 19
    data = torch.arange(20, dtype=torch.long)
    block_size = 4
    dataset = TextDataset(data, block_size=block_size)

    # Total length should be len(data) - block_size = 20 - 4 = 16
    assert len(dataset) == 16

    # Test first item
    x0, y0 = dataset[0]
    assert x0.shape == (block_size,)
    assert y0.shape == (block_size,)
    assert torch.equal(x0, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(y0, torch.tensor([1, 2, 3, 4]))

    # Target y should be shifted by 1 relative to x: y[t] == x[t+1] for 0 <= t < block_size - 1
    assert torch.equal(y0[:-1], x0[1:])

    # Test out of bounds index
    with pytest.raises(IndexError):
        _ = dataset[16]


def test_get_batch_shapes_and_properties():
    # Create synthetic token data
    data = torch.arange(1000, dtype=torch.long)
    block_size = 8
    batch_size = 4

    x, y = get_batch(data, block_size=block_size, batch_size=batch_size, device="cpu")

    # Check tensor dimensions: (Batch, Sequence Length) = (B, T)
    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)
    assert x.dtype == torch.long
    assert y.dtype == torch.long

    # Verify that in every row of the batch, y is shifted by exactly 1 position from x
    for b in range(batch_size):
        assert torch.equal(y[b, :-1], x[b, 1:])


def test_create_dataloader():
    data = torch.arange(100, dtype=torch.long)
    block_size = 10
    dataset = TextDataset(data, block_size=block_size)
    loader = create_dataloader(dataset, batch_size=4, shuffle=False)

    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (4, 10)
    assert batch_y.shape == (4, 10)


def test_prepare_corpus(tmp_path: Path):
    raw_file, train_file, val_file = prepare_corpus(output_dir=tmp_path, train_ratio=0.8)
    assert raw_file.exists()
    assert train_file.exists()
    assert val_file.exists()

    raw_content = raw_file.read_text(encoding="utf-8")
    train_content = train_file.read_text(encoding="utf-8")
    val_content = val_file.read_text(encoding="utf-8")

    assert len(train_content) > 0
    assert len(val_content) > 0
    assert len(train_content) > len(val_content)

