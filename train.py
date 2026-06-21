"""
Training script for MiniLLM on jtatman/python-code-dataset-500k.

Key choices:
1) Dataset fields are treated as instruction-following pairs:
   - prompt = optional system + instruction
   - target = output
2) GPT-4 / GPT-3.5 tokenizer: tiktoken "cl100k_base"
3) Prompt-masked causal LM training
4) Checkpoint save at the end

Run:
    python train.py

Environment overrides:
    DATASET_ID         default: jtatman/python-code-dataset-500k
    OUTPUT_DIR         default: ./checkpoints/mini_llm_python
    MAX_SEQ_LEN        default: 512
    BATCH_SIZE         default: 8
    GRAD_ACCUM_STEPS   default: 4
    NUM_EPOCHS         default: 1
    LR                 default: 3e-4
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset, load_from_disk, DatasetDict
import tiktoken

from mini_llm import MiniLLM, ModelConfig


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

DATASET_ID = os.environ.get("DATASET_ID", "jtatman/python-code-dataset-500k")
DATASET_PATH = os.environ.get("DATASET_PATH", "/home/anoush/Desktop/GPT/modern_GPT/dataset/python-code-dataset-500k").strip()  # optional local path
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./checkpoints"))

MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "1024"))  # includes prompt + answer
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", "4"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "2"))
LR = float(os.environ.get("LR", "3e-4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.1"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "200"))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", "1.0"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
DATASET_FRACTION = float(os.environ.get("DATASET_FRACTION", "0.02"))  # Use a fraction of the dataset for faster experiments (0 < fraction <= 1.0)

SEED = int(os.environ.get("SEED", "42"))





# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# -----------------------------------------------------------------------------
# Tokenizer
# -----------------------------------------------------------------------------

enc = tiktoken.get_encoding("cl100k_base")
VOCAB_SIZE = enc.n_vocab
EOS_ID = enc.eot_token  # special token id used as sequence terminator / pad token


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------

def _read_dataset():
    if DATASET_PATH:
        print("============================")
        ds = load_from_disk(DATASET_PATH)
        ds = ds["train"] if isinstance(ds, DatasetDict) else ds
    else:
        ds = load_dataset(DATASET_ID, split="train")

    if DATASET_FRACTION < 1.0:
        n = max(1, int(len(ds) * DATASET_FRACTION))
        ds = ds.shuffle(seed=SEED).select(range(n))
        print(f"Using {DATASET_FRACTION:.0%} of dataset → {n:,} examples")

    return ds
# def _read_dataset():
#     if DATASET_PATH:
#         ds = load_from_disk(DATASET_PATH)
#         return ds["train"] if isinstance(ds, DatasetDict) else ds
#     return load_dataset(DATASET_ID, split="train")

def _normalize_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_prompt(system: str, instruction: str) -> str:
    """
    Stable prompt template.

    We keep this simple on purpose:
    - system is optional
    - instruction is the task
    - response marks the generation target
    """
    parts: List[str] = []
    system = _normalize_text(system)
    instruction = _normalize_text(instruction)

    if system:
        parts.append(f"### System:\n{system}")
    parts.append(f"### Instruction:\n{instruction}")
    parts.append("### Response:\n")
    return "\n\n".join(parts)


def preprocess_example(example: Dict) -> Dict[str, List[int]]:
    """
    Converts one raw row into token ids.

    Returns:
      - input_ids: prompt + answer, shifted left by one for teacher forcing
      - labels: next-token targets, with prompt positions masked out
      - prompt_len: number of prompt tokens (for masking)
    """
    prompt = build_prompt(example.get("system", ""), example.get("instruction", ""))
    answer = _normalize_text(example.get("output", ""))

    prompt_ids = enc.encode(prompt, disallowed_special=())
    answer_ids = enc.encode(answer, disallowed_special=()) + [EOS_ID]

    tokens = prompt_ids + answer_ids
    if len(tokens) < 2:
        return {"input_ids": [], "labels": [], "prompt_len": 0}

    # Keep one extra token because we create (input_ids, labels) by next-token shift.
    max_tokens = MAX_SEQ_LEN + 1
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        prompt_len = min(len(prompt_ids), len(tokens) - 1)
    else:
        prompt_len = len(prompt_ids)

    input_ids = tokens[:-1]
    labels = tokens[1:]

    # Mask all prompt positions. The first answer token is predicted at index prompt_len - 1.
    mask_upto = max(prompt_len - 1, 0)
    for i in range(min(mask_upto, len(labels))):
        labels[i] = -100

    return {
        "input_ids": input_ids,
        "labels": labels,
        "prompt_len": prompt_len,
    }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Dynamic padding with a causal + key-padding attention mask.

    Shapes:
      input_ids: (B, T)
      labels:    (B, T)
      mask:      (B, 1, T, T) additive mask for SDPA
    """
    batch = [x for x in batch if x["input_ids"] and x["labels"]]
    if not batch:
        raise ValueError("Empty batch after preprocessing")

    max_len = min(max(len(x["input_ids"]) for x in batch), MAX_SEQ_LEN)

    input_ids_list = []
    labels_list = []
    valid_lengths = []

    for x in batch:
        inp = x["input_ids"][:max_len]
        lab = x["labels"][:max_len]

        valid_len = len(inp)
        pad_len = max_len - valid_len

        if pad_len > 0:
            inp = inp + [EOS_ID] * pad_len
            lab = lab + [-100] * pad_len

        input_ids_list.append(inp)
        labels_list.append(lab)
        valid_lengths.append(valid_len)

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    labels = torch.tensor(labels_list, dtype=torch.long)

    # Additive attention mask: 0 for allowed positions, -inf for masked positions.
    # We use a causal mask and additionally mask keys that are padding.
    causal = torch.full((max_len, max_len), float("-inf"))
    causal = torch.triu(causal, diagonal=1)

    masks = []
    for valid_len in valid_lengths:
        m = causal.clone()
        if valid_len < max_len:
            m[:, valid_len:] = float("-inf")
        masks.append(m)

    attn_mask = torch.stack(masks, dim=0).unsqueeze(1)  # (B, 1, T, T)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "mask": attn_mask,
    }


# -----------------------------------------------------------------------------
# Training utilities
# -----------------------------------------------------------------------------


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(model: MiniLLM, cfg: ModelConfig, tokenizer_name: str, out_dir: Path, name: str = "final.pt"):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / name

    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(cfg),
        "tokenizer_name": tokenizer_name,
        "tokenizer_vocab_size": VOCAB_SIZE,
        "eos_id": EOS_ID,
    }
    torch.save(payload, ckpt_path)

    meta_path = out_dir / "training_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_id": DATASET_ID,
                "dataset_path": DATASET_PATH,
                "tokenizer_name": tokenizer_name,
                "tokenizer_vocab_size": VOCAB_SIZE,
                "eos_id": EOS_ID,
                "max_seq_len": MAX_SEQ_LEN,
                "batch_size": BATCH_SIZE,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "num_epochs": NUM_EPOCHS,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
            },
            f,
            indent=2,
        )

    print(f"Saved checkpoint to: {ckpt_path}")
    print(f"Saved metadata to:   {meta_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    device = get_device()
    amp_dtype = get_amp_dtype(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))


    print(f"Device: {device}")
    print(f"AMP dtype: {amp_dtype}")
    print(f"Tokenizer: cl100k_base (vocab={VOCAB_SIZE}, eos={EOS_ID})")

    raw_ds = _read_dataset()
    print(raw_ds)

    columns = set(raw_ds.column_names)

    required = {"instruction", "output"}
    missing = required - columns
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Tokenize offline to keep the training loop simple and deterministic.
    tokenized_ds = raw_ds.map(
        preprocess_example,
        remove_columns=raw_ds.column_names,
        desc="Tokenizing",
    )

    tokenized_ds = tokenized_ds.filter(lambda x: len(x["input_ids"]) > 0 and len(x["labels"]) > 0)
    print(tokenized_ds)

    cfg = ModelConfig(
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
    )
    model = MiniLLM(cfg).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
    )

    loader = DataLoader(
        tokenized_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=True,
    )

    total_update_steps = max((len(loader) * NUM_EPOCHS) // GRAD_ACCUM_STEPS, 1)

    update_step = 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoints under: {OUTPUT_DIR}")

    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0

    for epoch in range(NUM_EPOCHS):
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                logits = model(input_ids, mask=mask)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
                loss = loss / GRAD_ACCUM_STEPS

            # loss.backward()
            scaler.scale(loss).backward()
            running_loss += loss.item()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                lr = cosine_warmup_lr(update_step, total_update_steps, WARMUP_STEPS, LR)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                # torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                # optimizer.step()
                # optimizer.zero_grad(set_to_none=True)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                update_step += 1

                if update_step % 10 == 0:
                    avg_loss = running_loss * GRAD_ACCUM_STEPS / 10.0
                    print(
                        f"epoch={epoch + 1}/{NUM_EPOCHS} "
                        f"step={update_step}/{total_update_steps} "
                        f"lr={lr:.2e} "
                        f"loss={avg_loss:.4f}"
                    )
                    running_loss = 0.0
            if (len(loader) % GRAD_ACCUM_STEPS) != 0:
                optimizer.zero_grad(set_to_none=True)
        try:
            print(f"Finished epoch {epoch + 1}/{NUM_EPOCHS} Loss: {avg_loss}")
        except:
            print(f"Finished epoch {epoch + 1}/{NUM_EPOCHS}")

    save_checkpoint(model, cfg, "cl100k_base", OUTPUT_DIR, name="mini_llm_final.pt")


if __name__ == "__main__":
    main()
