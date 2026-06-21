# MiniLLM

A from-scratch, decoder-only Transformer (~100M params) implementing modern LLaMA-style design choices, trained on Python code instructions.

## Architecture

`mini_llm.py` implements:

- **RoPE** — Rotary Position Embeddings
- **GQA** — Grouped Query Attention
- **Flash Attention** — via PyTorch SDPA
- **SwiGLU** — gated FFN activation
- **RMSNorm** — pre-norm, no mean subtraction
- **No bias** in any linear layer
- **Weight tying** — input embeddings shared with the LM head

Default config (~100M params): `d_model=768, num_heads=12, num_kv_heads=3, num_layers=12, d_ff=2048, vocab_size=32000` (tokenizer vocab is set automatically at training time — see below).

```python
from mini_llm import MiniLLM, ModelConfig

cfg = ModelConfig()
model = MiniLLM(cfg)
```

Want it bigger? Scale to ~1B with `d_model=2048, num_heads=16, num_kv_heads=4, num_layers=24`.

## Training

`train.py` fine-tunes MiniLLM on [jtatman/python-code-dataset-500k](https://huggingface.co/datasets/jtatman/python-code-dataset-500k) as instruction → code pairs.

- **Tokenizer**: `tiktoken` `cl100k_base` (GPT-3.5/4 tokenizer)
- **Objective**: prompt-masked causal LM (loss only computed on the response tokens)
- **Mixed precision**: bf16/fp16 autocast + grad scaler
- **LR schedule**: cosine decay with linear warmup
- **Gradient accumulation** + grad norm clipping

Run:

```bash
python train.py
```

Configure via environment variables:

| Variable | Default | Description |
|---|---|---|
| `DATASET_ID` | `jtatman/python-code-dataset-500k` | HF dataset to pull if no local path is set |
| `DATASET_PATH` | _(unset)_ | Optional local path to a pre-downloaded dataset |
| `OUTPUT_DIR` | `./checkpoints` | Where checkpoints are saved |
| `MAX_SEQ_LEN` | `1024` | Max tokens per example (prompt + answer) |
| `BATCH_SIZE` | `2` | Per-step batch size |
| `GRAD_ACCUM_STEPS` | `4` | Gradient accumulation steps |
| `NUM_EPOCHS` | `2` | Training epochs |
| `LR` | `3e-4` | Peak learning rate |
| `WEIGHT_DECAY` | `0.1` | AdamW weight decay |
| `WARMUP_STEPS` | `200` | LR warmup steps |
| `MAX_GRAD_NORM` | `1.0` | Gradient clipping threshold |
| `DATASET_FRACTION` | `0.02` | Fraction of dataset to use (for fast experiments) |
| `SEED` | `42` | Random seed |

A final checkpoint (`mini_llm_final.pt`) plus `training_meta.json` are written to `OUTPUT_DIR`.

## Project structure

```
.
├── mini_llm.py        # Model definition (config, layers, generate())
├── train.py            # Training loop
├── checkpoints/         # Saved model weights (empty until you train)
└── dataset/             # Optional local dataset cache (empty by default)
```

## Requirements

- Python 3.10+
- `torch >= 2.0`
- `tiktoken`
- `datasets`

```bash
pip install torch tiktoken datasets
```

## Generation

```python
import torch
from mini_llm import MiniLLM, ModelConfig

cfg = ModelConfig()
model = MiniLLM(cfg)
ckpt = torch.load("checkpoints/mini_llm_final.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])

out = model.generate(prompt_ids, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.9)
```

## Status

Educational / research project — built to understand and reproduce modern LLM architecture and training mechanics, not for production use.

## License

MIT
