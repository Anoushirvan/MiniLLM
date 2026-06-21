# test.py

import torch
import tiktoken

from mini_llm import MiniLLM, ModelConfig


CHECKPOINT = "./checkpoints/mini_llm_final.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# Load checkpoint
# --------------------------------------------------

ckpt = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

print("Checkpoint keys:")
print(ckpt.keys())


# --------------------------------------------------
# Restore config
# --------------------------------------------------

# cfg_dict = ckpt["config"]
cfg_dict = ckpt["model_config"]
cfg = ModelConfig(**cfg_dict)

model = MiniLLM(cfg)

model.load_state_dict(
    ckpt["model_state_dict"]
)

model.to(DEVICE)
model.eval()


# --------------------------------------------------
# Restore tokenizer
# --------------------------------------------------

tokenizer_name = ckpt.get(
    "tokenizer_name",
    "cl100k_base"
)

tokenizer = tiktoken.get_encoding(
    tokenizer_name
)

print(f"Tokenizer: {tokenizer_name}")


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def encode(text):
    return tokenizer.encode(text)


def decode(tokens):
    return tokenizer.decode(tokens)


# --------------------------------------------------
# Prompt
# --------------------------------------------------

instruction = """
Write a Python function that computes fibonacci numbers.
"""

prompt = f"""### Instruction:
{instruction}

### Response:
"""

input_ids = torch.tensor(
    [encode(prompt)],
    dtype=torch.long,
    device=DEVICE
)


# --------------------------------------------------
# Generate
# --------------------------------------------------

with torch.no_grad():

    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=256,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.1
    )

output_text = decode(
    output_ids[0].tolist()
)

print("\n")
print("=9=" * 80)
print(output_text)
print("=" * 80)