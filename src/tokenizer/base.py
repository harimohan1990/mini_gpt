"""Base tokenizer interface for Mini GPT."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional
import torch


class BaseTokenizer(ABC):
    """Abstract base class for all tokenizers in Mini GPT."""

    def __init__(
        self,
        unk_token: Optional[str] = "<unk>",
        pad_token: Optional[str] = "<pad>",
        bos_token: Optional[str] = "<bos>",
        eos_token: Optional[str] = "<eos>",
    ) -> None:
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Returns the total number of unique tokens in the vocabulary."""
        pass

    @property
    @abstractmethod
    def token_to_id(self) -> Dict[str, int]:
        """Returns the mapping from token strings to integer IDs."""
        pass

    @property
    @abstractmethod
    def id_to_token(self) -> Dict[int, str]:
        """Returns the mapping from integer IDs to token strings."""
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Converts a raw string into a list of integer token IDs."""
        pass

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Converts a list of integer token IDs back into a reconstructed string."""
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Serializes tokenizer vocabulary and state to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> BaseTokenizer:
        """Loads a serialized tokenizer from disk."""
        pass

    def encode_to_tensor(self, text: str, device: str = "cpu") -> torch.Tensor:
        """Encodes text and returns a 1D PyTorch tensor of dtype torch.long."""
        ids = self.encode(text)
        return torch.tensor(ids, dtype=torch.long, device=device)

    def decode_from_tensor(self, tensor: torch.Tensor) -> str:
        """Decodes a 1D PyTorch tensor or list of integers into a string."""
        if tensor.ndim > 1:
            tensor = tensor.squeeze()
        return self.decode(tensor.tolist())

