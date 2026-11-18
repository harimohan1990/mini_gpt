"""Byte Pair Encoding (BPE) Tokenizer for Mini GPT.

Implements from-scratch BPE tokenization:
1. Base vocabulary initialization from individual characters
2. Iterative pair frequency computation
3. Greedy merge selection and merge rule recording
4. Rank-based subword encoding
5. Concatenation decoding
6. JSON serialization and deserialization
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.tokenizer.base import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """Subword tokenizer based on the Byte Pair Encoding (BPE) algorithm."""

    def __init__(
        self,
        token_to_id: Dict[str, int],
        merges: List[Tuple[str, str]],
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
        self.merges = merges
        # Lookup table for quick rank resolution: lower rank = merged earlier
        self.bpe_ranks: Dict[Tuple[str, str], int] = {pair: i for i, pair in enumerate(merges)}

        self.unk_id = self._token_to_id.get(unk_token) if unk_token else None
        self.pad_id = self._token_to_id.get(pad_token) if pad_token else None
        self.bos_id = self._token_to_id.get(bos_token) if bos_token else None
        self.eos_id = self._token_to_id.get(eos_token) if eos_token else None

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    @property
    def token_to_id(self) -> Dict[str, int]:
        return self._token_to_id

    @property
    def id_to_token(self) -> Dict[int, str]:
        return self._id_to_token

    @staticmethod
    def get_pair_stats(corpus_words: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """Counts frequency of all adjacent token pairs across word frequencies."""
        stats: Dict[Tuple[str, str], int] = defaultdict(int)
        for word_tuple, freq in corpus_words.items():
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i + 1])
                stats[pair] += freq
        return stats

    @staticmethod
    def merge_vocab(
        pair: Tuple[str, str],
        corpus_words: Dict[Tuple[str, ...], int],
    ) -> Dict[Tuple[str, ...], int]:
        """Replaces all occurrences of pair (p0, p1) with single token (p0+p1)."""
        new_corpus: Dict[Tuple[str, ...], int] = {}
        bigram = pair
        replacement = "".join(pair)

        for word_tuple, freq in corpus_words.items():
            new_word: List[str] = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == bigram:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            new_corpus[tuple(new_word)] = freq
        return new_corpus

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        unk_token: Optional[str] = "<unk>",
        pad_token: Optional[str] = "<pad>",
        bos_token: Optional[str] = "<bos>",
        eos_token: Optional[str] = "<eos>",
    ) -> BPETokenizer:
        """Trains BPE merge rules on raw text until target vocab_size is reached."""
        special_tokens = [tok for tok in [unk_token, pad_token, bos_token, eos_token] if tok is not None]

        # Extract words from corpus and split into character tuples
        words = text.split()
        word_counts: Dict[str, int] = defaultdict(int)
        for w in words:
            word_counts[w] += 1

        # Format each word as a tuple of single characters
        corpus_words: Dict[Tuple[str, ...], int] = {
            tuple(list(w)): count for w, count in word_counts.items()
        }

        # Initialize base vocabulary with unique individual characters from corpus
        base_chars: Set[str] = set(text)
        # Ensure space is explicitly in base vocabulary
        base_chars.add(" ")
        # Exclude special tokens if they collide
        base_chars = {c for c in base_chars if c not in special_tokens}

        # Build initial vocab list
        vocab_list = special_tokens + sorted(list(base_chars))
        token_to_id = {tok: idx for idx, tok in enumerate(vocab_list)}

        num_merges = vocab_size - len(token_to_id)
        if num_merges < 0:
            raise ValueError(
                f"Requested vocab_size {vocab_size} is smaller than base alphabet size {len(token_to_id)}"
            )

        merges: List[Tuple[str, str]] = []

        # Iteratively merge the most frequent pair
        for _ in range(num_merges):
            stats = cls.get_pair_stats(corpus_words)
            if not stats:
                break  # No more pairs to merge

            # Pick pair with maximum frequency
            best_pair = max(stats, key=stats.get)  # type: ignore[arg-type]
            if stats[best_pair] < 1:
                break

            merges.append(best_pair)
            new_token = "".join(best_pair)
            token_to_id[new_token] = len(token_to_id)

            corpus_words = cls.merge_vocab(best_pair, corpus_words)

        return cls(
            token_to_id=token_to_id,
            merges=merges,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
        )

    def _bpe_tokenize_word(self, word: str) -> List[str]:
        """Tokenizes an individual word using learned merge ranks."""
        if not word:
            return []

        word_tuple = tuple(list(word))
        if len(word_tuple) <= 1:
            return list(word_tuple)

        while True:
            # Find all adjacent pairs
            pairs = [(word_tuple[i], word_tuple[i + 1]) for i in range(len(word_tuple) - 1)]
            # Find pair that occurs earliest in merge rules
            eligible_pairs = [p for p in pairs if p in self.bpe_ranks]
            if not eligible_pairs:
                break

            min_pair = min(eligible_pairs, key=lambda p: self.bpe_ranks[p])
            replacement = "".join(min_pair)

            new_word: List[str] = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == min_pair:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            word_tuple = tuple(new_word)
            if len(word_tuple) <= 1:
                break

        return list(word_tuple)

    def encode(self, text: str) -> List[int]:
        """Encodes string into subword token IDs."""
        words = text.split(" ")
        token_ids: List[int] = []

        for i, word in enumerate(words):
            if i > 0:
                # Add whitespace token if present, or encode space
                if " " in self._token_to_id:
                    token_ids.append(self._token_to_id[" "])
                elif self.unk_id is not None:
                    token_ids.append(self.unk_id)

            subwords = self._bpe_tokenize_word(word)
            for sw in subwords:
                if sw in self._token_to_id:
                    token_ids.append(self._token_to_id[sw])
                elif self.unk_id is not None:
                    token_ids.append(self.unk_id)
                else:
                    raise KeyError(f"Token '{sw}' cannot be encoded.")

        return token_ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decodes token IDs back into string."""
        specials = {self.unk_id, self.pad_id, self.bos_id, self.eos_id} - {None}
        parts: List[str] = []
        for tid in ids:
            if skip_special_tokens and tid in specials:
                continue
            if tid in self._id_to_token:
                parts.append(self._id_to_token[tid])
            elif self.unk_token:
                parts.append(self.unk_token)
        return "".join(parts)

    def save(self, path: str | Path) -> None:
        """Saves BPE tokenizer state to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": "bpe",
            "token_to_id": self._token_to_id,
            "merges": self.merges,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Loads BPE tokenizer from JSON."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        merges = [tuple(pair) for pair in data["merges"]]
        return cls(
            token_to_id=data["token_to_id"],
            merges=merges,
            unk_token=data.get("unk_token"),
            pad_token=data.get("pad_token"),
            bos_token=data.get("bos_token"),
            eos_token=data.get("eos_token"),
        )
