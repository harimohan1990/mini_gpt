"""Autoregressive text generation engine for Mini GPT.

Implements prompt encoding, step-by-step next token generation, sampling controls,
repetition mitigation, stop-sequence halting, and token streaming.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, List, Optional, Union
import torch

from src.model.gpt import MiniGPT, GPTConfig
from src.tokenizer.base import BaseTokenizer
from src.generation.sampler import (
    apply_repetition_penalty,
    sample_next_token,
)


class TextGenerator:
    """High-level text generation orchestrator for Mini GPT."""

    def __init__(
        self,
        model: MiniGPT,
        tokenizer: BaseTokenizer,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        tokenizer: BaseTokenizer,
        device: str = "auto",
    ) -> TextGenerator:
        """Instantiates TextGenerator directly from a saved checkpoint file."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        # Device resolution
        if device == "auto":
            if torch.cuda.is_available():
                chosen_device = "cuda"
            elif torch.backends.mps.is_available():
                chosen_device = "mps"
            else:
                chosen_device = "cpu"
        else:
            chosen_device = device

        checkpoint = torch.load(checkpoint_path, map_location=chosen_device, weights_only=False)
        config: GPTConfig = checkpoint["config"]
        model = MiniGPT(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(chosen_device)
        model.eval()

        return cls(model=model, tokenizer=tokenizer, device=chosen_device)

    @torch.no_grad()
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.8,
        top_k: Optional[int] = 20,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
        stop_sequences: Optional[List[str]] = None,
    ) -> Generator[str, None, str]:
        """Yields each generated token as it is produced (streaming mode).

        Returns the full concatenated continuation at completion.
        """
        # Encode initial prompt
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = [self.tokenizer.bos_id if self.tokenizer.bos_id is not None else 0]

        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        eos_id = self.tokenizer.eos_id
        generated_token_ids: List[int] = []
        full_continuation: List[str] = []

        for _ in range(max_new_tokens):
            # Crop prompt to context window if length exceeds block_size
            idx_cond = idx if idx.size(1) <= self.model.config.block_size else idx[:, -self.model.config.block_size :]

            # Forward pass: extract logits for the final token position
            logits, _ = self.model(idx_cond)
            logits_step = logits[:, -1, :].clone()  # Shape: (1, vocab_size)

            # Apply repetition penalty
            if repetition_penalty > 1.0 and (prompt_ids + generated_token_ids):
                all_context_tokens = prompt_ids + generated_token_ids
                logits_step = apply_repetition_penalty(
                    logits_step,
                    generated_tokens=all_context_tokens[-self.model.config.block_size :],
                    penalty=repetition_penalty,
                )

            # Sample next token index
            next_token_tensor = sample_next_token(
                logits=logits_step,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            next_token_id = int(next_token_tensor.item())

            # Stop condition 1: EOS token
            if eos_id is not None and next_token_id == eos_id:
                break

            generated_token_ids.append(next_token_id)
            idx = torch.cat((idx, next_token_tensor), dim=1)

            # Decode latest token string
            token_str = self.tokenizer.decode([next_token_id])
            full_continuation.append(token_str)
            yield token_str

            # Stop condition 2: Stop sequence match
            if stop_sequences:
                current_text = "".join(full_continuation)
                if any(stop_seq in current_text for stop_seq in stop_sequences):
                    break

        return "".join(full_continuation)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.8,
        top_k: Optional[int] = 20,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generates complete text continuation for a given prompt."""
        gen = self.stream_generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_sequences=stop_sequences,
        )
        tokens = list(gen)
        return "".join(tokens)

