"""FastAPI application for Mini GPT.

Provides:
1. Health and Model Metadata endpoints
2. Raw text generation endpoint
3. ChatGPT-style multi-turn conversation endpoint
4. Conversation history clearing
5. Static web UI serving
6. CORS and structured Pydantic error validation
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Literal, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from src.generation.generate import TextGenerator
from src.model.gpt import MiniGPT, GPTConfig
from src.tokenizer.char_tokenizer import CharTokenizer
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.tokenizer.base import BaseTokenizer


# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1, description="Content of the message")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1, description="List of conversation messages")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: Optional[int] = Field(20, ge=0, le=100, description="Top-K candidates")
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Top-P cumulative probability")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")
    max_tokens: int = Field(120, ge=1, le=500, description="Max new tokens to generate")


class ChatResponse(BaseModel):
    response: str
    tokens_generated: int
    generation_time_ms: float
    model: str = "Mini-GPT"


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input text prompt")
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(20, ge=0)
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    max_tokens: int = Field(120, ge=1, le=500)


class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    full_text: str
    tokens_generated: int
    generation_time_ms: float


class ModelInfoResponse(BaseModel):
    status: str
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    parameters: int
    device: str


# ---------------------------------------------------------
# Global App State & Lifespan
# ---------------------------------------------------------

class AppState:
    generator: Optional[TextGenerator] = None
    tokenizer: Optional[BaseTokenizer] = None
    model: Optional[MiniGPT] = None
    device: str = "cpu"


state = AppState()


def load_model_and_tokenizer(
    checkpoint_path: Path = Path("checkpoints/best_model.pt"),
    tokenizer_path: Path = Path("data/processed/char_tokenizer.json"),
) -> None:
    """Loads tokenizer and MiniGPT model into global state."""
    # 1. Load Tokenizer
    if tokenizer_path.exists():
        if "bpe" in str(tokenizer_path).lower():
            state.tokenizer = BPETokenizer.load(tokenizer_path)
        else:
            state.tokenizer = CharTokenizer.load(tokenizer_path)
    else:
        # Fallback character tokenizer for testing environments
        fallback_corpus = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\n:-,.?"
        state.tokenizer = CharTokenizer.build_from_corpus(fallback_corpus)

    # 2. Load Model Checkpoint
    if checkpoint_path.exists():
        state.generator = TextGenerator.from_checkpoint(
            checkpoint_path=checkpoint_path,
            tokenizer=state.tokenizer,
            device="auto",
        )
        state.model = state.generator.model
        state.device = state.generator.device
    else:
        # Fallback lightweight model instance for cold start / testing
        config = GPTConfig(
            vocab_size=state.tokenizer.vocab_size,
            block_size=64,
            n_layer=2,
            n_head=2,
            n_embd=64,
        )
        state.model = MiniGPT(config)
        state.device = "cpu"
        state.generator = TextGenerator(
            model=state.model,
            tokenizer=state.tokenizer,
            device="cpu",
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager initializing model artifacts at startup."""
    load_model_and_tokenizer()
    yield


app = FastAPI(
    title="Mini GPT API",
    description="ChatGPT-like REST interface for Mini GPT built from scratch in PyTorch.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def format_conversation_prompt(messages: List[ChatMessage]) -> str:
    """Formats multi-turn chat messages into the model's dialogue representation.

    Format:
        ALICE: <user message>
        BOB: <assistant message>
        ALICE: <next user message>
        BOB:
    """
    formatted_parts: List[str] = []
    for msg in messages:
        if msg.role == "user":
            formatted_parts.append(f"ALICE: {msg.content.strip()}\n")
        elif msg.role == "assistant":
            formatted_parts.append(f"BOB: {msg.content.strip()}\n")
        elif msg.role == "system":
            formatted_parts.append(f"THEORY: {msg.content.strip()}\n")

    # Prompt assistant turn
    formatted_parts.append("BOB: ")
    return "".join(formatted_parts)


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, str]:
    """Health check endpoint verifying model readiness."""
    if state.generator is None or state.model is None:
        return {"status": "unhealthy", "model": "not_loaded"}
    return {
        "status": "healthy",
        "model": "Mini-GPT",
        "device": state.device,
    }


@app.get("/api/model-info", response_model=ModelInfoResponse)
def get_model_info() -> ModelInfoResponse:
    """Returns architecture specifications and parameter counts."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    return ModelInfoResponse(
        status="ready",
        vocab_size=state.model.config.vocab_size,
        block_size=state.model.config.block_size,
        n_layer=state.model.config.n_layer,
        n_head=state.model.config.n_head,
        n_embd=state.model.config.n_embd,
        parameters=state.model.get_num_params(non_embedding=False),
        device=state.device,
    )


@app.post("/api/generate", response_model=GenerateResponse)
def generate_text(req: GenerateRequest) -> GenerateResponse:
    """Generates text continuation conditioned on a raw prompt."""
    if state.generator is None or state.tokenizer is None:
        raise HTTPException(status_code=503, detail="Model generator is not loaded.")

    t0 = time.time()
    try:
        completion = state.generator.generate(
            prompt=req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            stop_sequences=["\n\n\n", "---"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    elapsed_ms = (time.time() - t0) * 1000.0
    tokens_generated = len(state.tokenizer.encode(completion))

    return GenerateResponse(
        prompt=req.prompt,
        completion=completion,
        full_text=req.prompt + completion,
        tokens_generated=tokens_generated,
        generation_time_ms=round(elapsed_ms, 2),
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat_turn(req: ChatRequest) -> ChatResponse:
    """Processes multi-turn chat messages and returns the assistant's reply."""
    if state.generator is None or state.tokenizer is None:
        raise HTTPException(status_code=503, detail="Model generator is not loaded.")

    # 1. Format multi-turn conversation into prompt string
    prompt = format_conversation_prompt(req.messages)

    # 2. Run autoregressive generation
    t0 = time.time()
    try:
        reply = state.generator.generate(
            prompt=prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            stop_sequences=["\nALICE:", "\nTHEORY:", "\n---"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")

    elapsed_ms = (time.time() - t0) * 1000.0

    # Clean reply: strip trailing speaker turns if generated
    cleaned_reply = reply.split("\nALICE:")[0].split("\nTHEORY:")[0].strip()
    if not cleaned_reply:
        cleaned_reply = reply.strip()

    tokens_generated = len(state.tokenizer.encode(cleaned_reply))

    return ChatResponse(
        response=cleaned_reply,
        tokens_generated=tokens_generated,
        generation_time_ms=round(elapsed_ms, 2),
    )


@app.post("/api/clear")
def clear_conversation() -> Dict[str, str]:
    """Endpoint confirming conversation state reset."""
    return {"status": "cleared", "message": "Conversation history reset successfully."}


# ---------------------------------------------------------
# Static Frontend Serving
# ---------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def serve_index() -> FileResponse:
    """Serves the ChatGPT-like React single page application."""
    frontend_path = Path("frontend/index.html")
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return HTMLResponse("<h1>Mini GPT API is running. Build frontend/index.html to view UI.</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)

