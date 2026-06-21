"""
MiniLLM — Decoder-only Transformer (~100M params)
══════════════════════════════════════════════════
Architecture mirrors LLaMA 2/3 design decisions:

    ✓  RoPE          — Rotary Position Embeddings
    ✓  GQA           — Grouped Query Attention
    ✓  Flash Attn    — via PyTorch SDPA (scaled_dot_product_attention)
    ✓  SwiGLU        — Gated FFN activation
    ✓  RMSNorm       — Pre-norm, no mean subtraction
    ✓  No bias       — In any linear layer
    ✓  Weight tying  — embed_in == lm_head

Requirements: torch >= 2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════ #
#  Config                                                                #
# ══════════════════════════════════════════════════════════════════════ #

@dataclass
class ModelConfig:
    # ── Vocabulary ────────────────────────────────────────────────── #
    vocab_size:     int   = 32_000

    # ── Dimensions ───────────────────────────────────────────────── #
    d_model:        int   = 768       # Hidden / embedding size
    num_heads:      int   = 8        # Query heads
    num_kv_heads:   int   = 2         # GQA: K/V heads  (num_heads % num_kv_heads == 0)
    num_layers:     int   = 8
    d_ff:           int   = None      # Auto: ⌈8/3 · d_model⌉ rounded to 256-multiple

    # ── Sequence ─────────────────────────────────────────────────── #
    max_seq_len:    int   = 1024

    # ── Regularisation ───────────────────────────────────────────── #
    dropout:        float = 0.0       # 0.0 standard for large-scale training

    # ── Norm & Positional ────────────────────────────────────────── #
    norm_eps:       float = 1e-6
    rope_theta:     float = 10_000.0  # YaRN / LongRoPE use larger values (e.g. 500_000)

    # ── Training tricks ──────────────────────────────────────────── #
    tie_embeddings: bool  = True      # Saves vocab_size × d_model params

    def __post_init__(self):
        assert self.d_model % self.num_heads == 0,    "d_model must be divisible by num_heads"
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        if self.d_ff is None:
            raw      = int(8 / 3 * self.d_model)
            self.d_ff = (raw + 255) // 256 * 256      # round up to nearest multiple of 256

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def num_groups(self) -> int:
        """How many query heads share one K/V head."""
        return self.num_heads // self.num_kv_heads


# ══════════════════════════════════════════════════════════════════════ #
#  RMSNorm                                                               #
# ══════════════════════════════════════════════════════════════════════ #

class RMSNorm(nn.Module):
    """
    LayerNorm minus the mean subtraction:
        RMSNorm(x) = x / RMS(x) · γ   where RMS(x) = √(mean(x²) + ε)

    Analogy: LayerNorm is like standardising to mean=0, std=1.
             RMSNorm only normalises scale — cheaper and equally stable.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rsqrt = 1 / √(mean(x²) + ε)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ══════════════════════════════════════════════════════════════════════ #
#  RoPE — Rotary Position Embeddings                                     #
# ══════════════════════════════════════════════════════════════════════ #

def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10_000.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Precomputes complex-valued frequency tensor: shape (max_seq_len, head_dim // 2)

    Analogy: each pair of embedding dimensions gets its own clock.
    Dimension pair 0 ticks fast (captures local patterns),
    pair d/2-1 ticks slow (captures global patterns).
    Two tokens at similar positions have nearly identical phases → high attention.

    Returns: complex64 tensor
    """
    assert head_dim % 2 == 0
    # Frequency for each dimension pair: θ_i = 1 / (theta ^ (2i / d))
    i      = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    freqs  = 1.0 / (theta ** (i / head_dim))              # (head_dim/2,)

    # Positions
    t      = torch.arange(max_seq_len, dtype=torch.float32, device=device)

    # Outer product → angle matrix: (T, head_dim/2)
    angles = torch.outer(t, freqs)

    # Represent as unit complex numbers e^(i·θ)
    return torch.polar(torch.ones_like(angles), angles)   # complex64


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Rotate Q or K using precomputed RoPE frequencies.

    Args:
        x         : (B, H, T, head_dim)  — float16/bfloat16/float32
        freqs_cis : (T, head_dim // 2)   — complex64, sliced to x's T

    The rotation is applied pair-wise: [x0, x1] → rotated by θ_0,
    [x2, x3] → rotated by θ_1, etc.
    """
    # Cast to float32, reshape to pairs, view as complex
    x_complex = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], -1, 2)
    )                                                      # (B, H, T, head_dim/2) complex

    # Broadcast freqs: (1, 1, T, head_dim/2)
    f = freqs_cis[:x.shape[2]].unsqueeze(0).unsqueeze(0)

    # Complex multiplication = rotation
    x_rotated = torch.view_as_real(x_complex * f).flatten(3)

    return x_rotated.to(x.dtype)                          # cast back to original dtype


# ══════════════════════════════════════════════════════════════════════ #
#  GQA + Flash Attention                                                 #
# ══════════════════════════════════════════════════════════════════════ #

class GQAFlashAttention(nn.Module):
    """
    Grouped Query Attention with Flash Attention (PyTorch SDPA).

    GQA analogy:
        Full MHA   → every employee has their own private filing cabinet (K/V).
        MQA        → everyone shares ONE filing cabinet (too much contention).
        GQA        → small groups share a cabinet — sweet spot of speed vs quality.
                        LLaMA 3 uses 8 Q-heads per K/V head.

    Flash Attention analogy:
        Naive attention writes the full (T×T) attention matrix to HBM (slow GPU RAM).
        Flash Attention tiles the computation to stay in SRAM (fast on-chip cache)
        — identical math, 3–8× faster, O(T) memory instead of O(T²).
        PyTorch SDPA (≥2.0) selects Flash Attention automatically when available.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads    = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.num_groups   = cfg.num_groups
        self.head_dim     = cfg.head_dim
        self.dropout      = cfg.dropout

        d, hkv, hd = cfg.d_model, cfg.num_kv_heads, cfg.head_dim

        # Q projects to full num_heads; K, V project to num_kv_heads only (GQA savings)
        self.W_q = nn.Linear(d, cfg.num_heads * hd, bias=False)
        self.W_k = nn.Linear(d, hkv * hd,           bias=False)
        self.W_v = nn.Linear(d, hkv * hd,           bias=False)
        self.W_o = nn.Linear(d, d,                  bias=False)

    def forward(
        self,
        x:         torch.Tensor,          # (B, T, d_model)
        freqs_cis: torch.Tensor,          # (T, head_dim // 2) — complex
        mask:      torch.Tensor | None = None,
    ) -> torch.Tensor:

        B, T, _ = x.shape

        # ── Linear projections ───────────────────────────────────── #
        Q = self.W_q(x).view(B, T, self.num_heads,    self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # Q: (B, num_heads, T, head_dim)
        # K: (B, num_kv_heads, T, head_dim)

        # ── Apply RoPE to Q and K (never V) ─────────────────────── #
        Q = apply_rope(Q, freqs_cis)
        K = apply_rope(K, freqs_cis)

        # ── Expand K and V to match Q's num_heads (GQA broadcast) ── #
        # (B, num_kv_heads, T, head_dim) → (B, num_heads, T, head_dim)
        # repeat_interleave: [h0, h1] with 4 groups → [h0, h0, h0, h0, h1, h1, h1, h1]
        if self.num_groups > 1:
            K = K.repeat_interleave(self.num_groups, dim=1)
            V = V.repeat_interleave(self.num_groups, dim=1)

        # ── Flash Attention via PyTorch SDPA ─────────────────────── #
        # PyTorch automatically selects Flash Attention 2 kernel when:
        #   • CUDA device  • dtype in {float16, bfloat16}  • no custom mask
        # is_causal=True injects the causal mask without materialising T×T matrix
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask = mask,
            dropout_p = self.dropout if self.training else 0.0,
            is_causal = (mask is None),  # use built-in causal mask when no custom mask
        )
        # out: (B, num_heads, T, head_dim)

        # ── Merge heads → output projection ─────────────────────── #
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_model)
        return self.W_o(out)


# ══════════════════════════════════════════════════════════════════════ #
#  SwiGLU Feed-Forward Network                                           #
# ══════════════════════════════════════════════════════════════════════ #

class SwiGLUFFN(nn.Module):
    """
    FFN(x) = down( SiLU(gate(x)) ⊗ up(x) )

    Analogy: two parallel streams.
      gate(x) → SiLU → acts as a learned filter (what to suppress)
      up(x)   → carries the actual content
      Element-wise product: content passes only where the gate allows it.

    Outperforms vanilla GELU FFN empirically. Uses 3 weight matrices
    vs 2, but d_ff is reduced (8/3 × d_model vs 4 × d_model) to keep
    param count similar.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up   = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff,   cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ══════════════════════════════════════════════════════════════════════ #
#  Transformer Block (Decoder Layer)                                     #
# ══════════════════════════════════════════════════════════════════════ #

class TransformerBlock(nn.Module):
    """
    Pre-norm decoder block — the core repeating unit.

        x = x + Attention( RMSNorm(x) )    # self-attention residual
        x = x + FFN( RMSNorm(x) )          # feed-forward residual

    Pre-norm vs post-norm analogy:
        Post-norm: run the block, then normalise the output.
        Pre-norm:  normalise first, then run the block.
        Pre-norm allows gradients to flow back through the residual stream
        unimpeded → more stable training at depth (>12 layers).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn      = GQAFlashAttention(cfg)
        self.ffn       = SwiGLUFFN(cfg)
        self.norm_attn = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.norm_ffn  = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.drop      = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x:         torch.Tensor,
        freqs_cis: torch.Tensor,
        mask:      torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm_attn(x), freqs_cis, mask))
        x = x + self.drop(self.ffn(self.norm_ffn(x)))
        return x


# ══════════════════════════════════════════════════════════════════════ #
#  MiniLLM — Full Decoder-Only Model                                     #
# ══════════════════════════════════════════════════════════════════════ #

class MiniLLM(nn.Module):
    """
    Decoder-only LLM stack.

    Default config (~100M parameters):
        d_model=768, num_heads=12, num_kv_heads=3
        num_layers=12, d_ff=2048, vocab_size=32000

    To scale up to ~1B: d_model=2048, num_heads=16, num_kv_heads=4, num_layers=24
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # ── Token embedding ─────────────────────────────────────── #
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # ── Transformer layers ──────────────────────────────────── #
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.num_layers)
        ])

        # ── Final norm + LM head ─────────────────────────────────── #
        self.norm_final = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head    = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: share embedding and output projection weights
        # Analogy: the model learns a single "token space" — decoding is
        # just finding which token your hidden state is closest to.
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # ── Precompute RoPE frequencies (registered as non-persistent buffer) #
        freqs = precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("freqs_cis", freqs, persistent=False)

        # ── Weight initialisation ────────────────────────────────── #
        self._init_weights()

    # ---------------------------------------------------------------- #

    def _init_weights(self):
        """
        GPT-2 style initialisation.
        Residual projection weights (W_o, down) are scaled by 1/√(2·L)
        to prevent variance explosion as the residual stream accumulates
        across L layers.
        """
        std = 0.02
        for name, p in self.named_parameters():
            if p.dim() < 2:
                continue                                # norms & biases
            if "W_o" in name or "down" in name:
                # Residual projections: scaled init
                nn.init.normal_(p, mean=0.0,
                                std=std / math.sqrt(2 * self.cfg.num_layers))
            else:
                nn.init.normal_(p, mean=0.0, std=std)

    # ---------------------------------------------------------------- #

    def forward(
        self,
        input_ids: torch.Tensor,          # (B, T)  — token indices
        mask:      torch.Tensor | None = None,
    ) -> torch.Tensor:                    # returns logits (B, T, vocab_size)

        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"

        x      = self.embed(input_ids)           # (B, T, d_model)
        freqs  = self.freqs_cis[:T]              # slice to current T

        for block in self.blocks:
            x = block(x, freqs, mask)

        x      = self.norm_final(x)
        logits = self.lm_head(x)                 # (B, T, vocab_size)
        return logits

    # ---------------------------------------------------------------- #

    @torch.inference_mode()
    def generate(
        self,
        input_ids:      torch.Tensor,     # (B, T_prompt) — initial token ids
        max_new_tokens: int   = 100,
        temperature:    float = 1.0,      # > 1 = creative, < 1 = conservative
        top_k:          int   = 50,       # 0 = disabled
        top_p:          float = 0.9,      # 1.0 = disabled (nucleus sampling)
        repetition_penalty: float = 1.0,  # > 1 penalises already-seen tokens
    ) -> torch.Tensor:
        """
        Autoregressive generation with top-k + nucleus (top-p) sampling.

        Sampling pipeline analogy (applied in order):
          1. temperature  → rescale logit confidence (high T = flat distribution)
          2. top-k        → hard cutoff: only keep the K best tokens
          3. top-p        → soft cutoff: keep smallest set whose cumulative prob ≥ p
          4. softmax + multinomial sample
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Crop to context window
            ids    = input_ids[:, -self.cfg.max_seq_len:]
            logits = self(ids)[:, -1, :]           # (B, vocab_size) — last position only

            # ── Repetition penalty ───────────────────────────────── #
            if repetition_penalty != 1.0:
                for b in range(ids.shape[0]):
                    for token_id in ids[b].unique():
                        logits[b, token_id] /= repetition_penalty

            # ── Temperature ─────────────────────────────────────── #
            logits = logits / max(temperature, 1e-5)

            # ── Top-k ───────────────────────────────────────────── #
            if top_k > 0:
                k          = min(top_k, logits.size(-1))
                threshold  = logits.topk(k, dim=-1).values[:, -1, None]
                logits     = logits.masked_fill(logits < threshold, float("-inf"))

            # ── Top-p (nucleus sampling) ─────────────────────────── #
            if top_p < 1.0:
                probs            = F.softmax(logits, dim=-1)
                sorted_p, sort_i = probs.sort(dim=-1, descending=True)
                cum_p            = sorted_p.cumsum(dim=-1)
                # Remove tokens beyond the nucleus
                remove           = (cum_p - sorted_p) > top_p
                sorted_p[remove] = 0.0
                sorted_p        /= sorted_p.sum(dim=-1, keepdim=True)
                # Scatter back to original token order
                probs            = torch.zeros_like(probs).scatter_(1, sort_i, sorted_p)
            else:
                probs = F.softmax(logits, dim=-1)

            # ── Sample ──────────────────────────────────────────── #
            next_id   = torch.multinomial(probs, num_samples=1)   # (B, 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

        return input_ids

    # ---------------------------------------------------------------- #

    def param_count(self) -> dict[str, int]:
        """Returns parameter counts by component."""
        counts = {}
        for name, p in self.named_parameters():
            component = name.split(".")[0]
            counts[component] = counts.get(component, 0) + p.numel()
        total = sum(p.numel() for p in self.parameters())
        counts["TOTAL"] = total
        return counts


# ══════════════════════════════════════════════════════════════════════ #
#  Sanity check & param summary                                          #
# ══════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── Build model ─────────────────────────────────────────────── #
    cfg   = ModelConfig()
    model = MiniLLM(cfg).to(device)

    # ── Param breakdown ──────────────────────────────────────────── #
    counts = model.param_count()
    print("──────── Parameter Breakdown ─────────────")
    for k, v in counts.items():
        tag = " ◄ TOTAL" if k == "TOTAL" else ""
        print(f"     {k:<15} {v / 1e6:>7.2f}M{tag}")
    print("──────────────────────────────────────────")

    # ── Forward pass ─────────────────────────────────────────────── #
    B, T  = 2, 256
    ids   = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits = model(ids)

    print(f"Input  shape : {ids.shape}")
    print(f"Logits shape : {logits.shape}")    # (2, 256, 32000)
    print()

    # ── Generation ───────────────────────────────────────────────── #
    prompt = torch.randint(0, cfg.vocab_size, (1, 16), device=device)
    out    = model.generate(
        prompt,
        max_new_tokens     = 32,
        temperature        = 0.8,
        top_k              = 50,
        top_p              = 0.9,
        repetition_penalty = 1.1,
    )
    print(f"Prompt length   : {prompt.shape[1]}")
    print(f"Generated length: {out.shape[1]}")  # 16 + 32 = 48



