# Mini GPT — A GPT-like Language Model from Scratch

A fully functional, decoder-only Generative Pre-trained Transformer (GPT) implemented from mathematical first principles in PyTorch.

---

## 🚀 Quick Start: Launch the ChatGPT Web App

The fastest way to experience Mini GPT is via the interactive **React + FastAPI Web Application**:

```bash
# Start the FastAPI server (serves both API and the React chat frontend)
python3 -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Now open your browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

You will see the full dark-mode ChatGPT interface where you can:
- Chat with Mini GPT in real-time.
- Tune **Temperature**, **Top-K**, **Top-P (Nucleus)**, and **Repetition Penalty** using live sliders.
- View real-time generation latency (ms) and throughput (tokens/sec).
- Clear conversation history.

---

## 💻 CLI Commands & Usage

### 1. Interactive Text Generation (`generate.py`)
Generate text directly in your terminal from a prompt with custom sampling controls:

```bash
# Basic generation
python3 generate.py --prompt "ALICE: What is the Transformer architecture?"

# Advanced sampling controls
python3 generate.py \
  --prompt "THEORY: Deep learning models rely on" \
  --max_tokens 150 \
  --temperature 0.7 \
  --top_k 20 \
  --top_p 0.9 \
  --repetition_penalty 1.15
```

### 2. Pre-Training from Scratch (`train.py`)
Train Mini GPT on the processed dataset (or your own custom text):

```bash
# Train on CPU or Apple Silicon Metal (MPS) / CUDA
python3 train.py \
  --max_iters 500 \
  --batch_size 32 \
  --block_size 128 \
  --n_layer 4 \
  --n_head 4 \
  --n_embd 128 \
  --lr 5e-4
```
Checkpoints are automatically saved to `checkpoints/best_model.pt` and `checkpoints/latest_model.pt`.

### 3. Model Evaluation (`evaluate.py`)
Compute research evaluation metrics on validation data:

```bash
python3 evaluate.py --checkpoint checkpoints/best_model.pt
```
Outputs:
- **Cross-Entropy Loss** and **Perplexity (PPL)**
- **Bits-Per-Character (BPC)** compression efficiency
- **Top-1, Top-3, and Top-5 Token Prediction Accuracy**
- **Distinct-1 & Distinct-2 Lexical Diversity Ratios**
- **Model Parameter Inventory & Memory Footprint (FP32, FP16, INT8)**

### 4. Hardware & Optimization Benchmarking (`benchmark.py`)
Benchmark inference latency, attention acceleration, and INT8 quantization:

```bash
python3 benchmark.py
```

### 5. Running the Complete Test Suite
Run all 58 automated unit tests across all 9 implementation phases:

```bash
python3 -m pytest tests/ -v
```

---

## 🐍 Python Programmatic Usage

You can easily import and use Mini GPT in your own Python scripts:

```python
import torch
from src.generation.generate import TextGenerator
from src.tokenizer.char_tokenizer import CharTokenizer

# 1. Load Tokenizer
tokenizer = CharTokenizer.load("data/processed/char_tokenizer.json")

# 2. Load Model from Checkpoint
generator = TextGenerator.from_checkpoint(
    checkpoint_path="checkpoints/best_model.pt",
    tokenizer=tokenizer,
    device="auto",  # Automatically selects 'mps', 'cuda', or 'cpu'
)

# 3. Generate Text
continuation = generator.generate(
    prompt="ALICE: What is self-attention?\nBOB: ",
    max_new_tokens=100,
    temperature=0.7,
    top_k=20,
    top_p=0.9,
    repetition_penalty=1.1,
)

print(continuation)
```

### Token Streaming Generator:
```python
for token in generator.stream_generate(prompt="THEORY: ", max_new_tokens=50):
    print(token, end="", flush=True)
```

---

## 📁 Project Architecture

```
minigpty/
├── data/
│   ├── raw/                 # input.txt (Raw text corpus)
│   └── processed/           # train.txt, val.txt, char_tokenizer.json, bpe_tokenizer.json
├── checkpoints/             # Trained model weights (best_model.pt, latest_model.pt)
├── src/
│   ├── dataset/             # Dataset ingestion, normalization, and autoregressive batching
│   ├── tokenizer/           # CharTokenizer and Byte Pair Encoding (BPE) implementations
│   ├── model/               # GPT decoder-only Transformer layers (Embeddings, Attention, FFN, Block, MiniGPT)
│   ├── training/            # CustomAdamW, cosine learning rate scheduler, loss functions, and Trainer
│   ├── generation/          # Sampler (Temperature, Top-K, Top-P, Repetition Penalty) and TextGenerator
│   ├── evaluation/          # Accuracy, Perplexity, Bits-per-token, Distinct-N, and ModelEvaluator
│   ├── api/                 # FastAPI backend with multi-turn chat and static file serving
│   └── optimization/        # Dynamic INT8 quantization and hardware latency benchmark
├── frontend/                # React ChatGPT single-page application (index.html)
├── tests/                   # 58 comprehensive unit tests
├── train.py                 # CLI training script
├── generate.py              # CLI text generation script
├── evaluate.py              # CLI research evaluation script
├── benchmark.py             # CLI optimization profiler
├── requirements.txt         # Dependencies
└── README.md
```

