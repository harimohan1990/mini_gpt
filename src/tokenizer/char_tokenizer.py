"""Character-level tokenizer for Mini GPT.

Implements discrete character mapping, deterministic vocabulary creation,
special tokens handling, and JSON serialization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.tokenizer.base import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    """Character-level tokenizer.

    Each character in the input string is treated as a discrete token.
    This provides an intuitive, deterministic foundation for understanding
    tokenization without the complexity of subword segmentation.
    """

    def __init__(
        self,
        token_to_id: Dict[str, int],
        unk_token: Optional[str] = "<unk>",
        pad_token: Optional[str] = "<pad>",
        bos_token: Optional[str] = "<bos>",
        eos_token: Optional[str] = "<eos>",
    ) -> None:
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
        )
        self._token_to_id = dict(token_to_id)
        self._id_to_token = {v: k for k, v in self._token_to_id.items()}

        if len(self._token_to_id) != len(self._id_to_token):
            raise ValueError("Token to ID mapping must be strictly bijective.")

        # Cache special token IDs
        self.unk_id = self._token_to_id.get(unk_token) if unk_token else None
        self.pad_id = self._token_to_id.get(pad_token) if pad_token else None
        self.bos_id = self._token_to_id.get(bos_token) if bos_token else None
        self.eos_id = self._token_to_id.get(eos_token) if eos_token else None

    @property
    def vocab_size(self) -> int:
        """Returns the total vocabulary size |V|."""
        return len(self._token_to_id)

    @property
    def token_to_id(self) -> Dict[str, int]:
        return self._token_to_id

    @property
    def id_to_token(self) -> Dict[int, str]:
        return self._id_to_token

    @classmethod
    def build_from_corpus(
        cls,
        text: str,
        unk_token: Optional[str] = "<unk>",
        pad_token: Optional[str] = "<pad>",
        bos_token: Optional[str] = "<bos>",
        eos_token: Optional[str] = "<eos>",
    ) -> CharTokenizer:
        """Constructs a character vocabulary from raw corpus text.

        The vocabulary starts with requested special tokens followed by all
        unique characters extracted from the corpus, sorted deterministically.
        """
        special_tokens = [tok for tok in [unk_token, pad_token, bos_token, eos_token] if tok is not None]
        unique_chars = sorted(list(set(text)))

        # Remove special tokens from unique_chars if they accidentally appear as characters
        unique_chars = [c for c in unique_chars if c not in special_tokens]

        # Assemble unified deterministic vocabulary
        vocab_list = special_tokens + unique_chars
        token_to_id = {token: idx for idx, token in enumerate(vocab_list)}

        return cls(
            token_to_id=token_to_id,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
        )

    def encode(self, text: str) -> List[int]:
        """Encodes a string of characters into integer IDs.

        If a character is not in the vocabulary, it maps to unk_id (if configured),
        or raises a KeyError.
        """
        encoded_ids: List[int] = []
        for char in text:
            if char in self._token_to_id:
                encoded_ids.append(self._token_to_id[char])
            elif self.unk_id is not None:
                encoded_ids.append(self.unk_id)
            else:
                raise KeyError(f"Character '{char}' not found in vocabulary and no unk_token defined.")
        return encoded_ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decodes a list of integer IDs back into a character string."""
        specials = {
            self.unk_id,
            self.pad_id,
            self.bos_id,
            self.eos_id,
        } - {None}

        chars: List[str] = []
        for token_id in ids:
            if skip_special_tokens and token_id in specials:
                continue
            if token_id in self._id_to_token:
                chars.append(self._id_to_token[token_id])
            else:
                chars.append(self.unk_token if self.unk_token else "")
        return "".join(chars)

    def save(self, path: str | Path) -> None:
        """Serializes tokenizer to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": "char",
            "token_to_id": self._token_to_id,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        """Loads a serialized tokenizer from a JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            token_to_id=data["token_to_id"],
            unk_token=data.get("unk_token"),
            pad_token=data.get("pad_token"),
            bos_token=data.get("bos_token"),
            eos_token=data.get("eos_token"),
        )

