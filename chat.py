"""Interactive terminal chat CLI for Mini GPT."""

from __future__ import annotations

import sys
from pathlib import Path

from src.generation.generate import TextGenerator
from src.tokenizer.char_tokenizer import CharTokenizer
from src.tokenizer.bpe_tokenizer import BPETokenizer


def main() -> None:
    ckpt_path = Path("checkpoints/best_model.pt")
    tok_path = Path("data/processed/char_tokenizer.json")

    if not ckpt_path.exists():
        print(f"Error: Model checkpoint not found at {ckpt_path}. Run 'python3 train.py' first.")
        sys.exit(1)

    # Load tokenizer & generator
    if "bpe" in str(tok_path).lower():
        tokenizer = BPETokenizer.load(tok_path)
    else:
        tokenizer = CharTokenizer.load(tok_path)

    generator = TextGenerator.from_checkpoint(checkpoint_path=ckpt_path, tokenizer=tokenizer, device="auto")

    print("\n" + "=" * 55)
    print("         MINI GPT INTERACTIVE TERMINAL CHAT")
    print("=" * 55)
    print(f"Model loaded on: {generator.device.upper()}")
    print("Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    conversation_history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            conversation_history.append(f"ALICE: {user_input}")

            # Keep last 3 turns to fit in context window
            recent_turns = conversation_history[-4:]
            prompt = "\n".join(recent_turns) + "\nBOB: "

            print("Mini GPT: ", end="", flush=True)
            response_chunks = []
            for token_chunk in generator.stream_generate(
                prompt=prompt,
                max_new_tokens=100,
                temperature=0.7,
                top_k=20,
                top_p=0.9,
                repetition_penalty=1.15,
                stop_sequences=["\nALICE:", "\nTHEORY:", "\n---"],
            ):
                print(token_chunk, end="", flush=True)
                response_chunks.append(token_chunk)
            print()

            assistant_reply = "".join(response_chunks).split("\nALICE:")[0].strip()
            conversation_history.append(f"BOB: {assistant_reply}")

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended. Goodbye!")
            break


if __name__ == "__main__":
    main()

