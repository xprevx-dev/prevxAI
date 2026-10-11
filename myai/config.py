"""Model and training configuration.

Presets are sized deliberately: `rtx3050_4gb` is tuned so a training run fits in
4 GB of VRAM with bf16 + gradient accumulation. See README "Sizing" for how to
pick one, or let `pick_preset_for_vram()` choose from the GPU it finds.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Optional


@dataclass
class ModelConfig:
    """Architecture of the decoder-only transformer."""

    vocab_size: int = 4096
    block_size: int = 512          # context window, in tokens
    n_layer: int = 4               # transformer blocks
    n_head: int = 4                # query heads
    n_kv_head: Optional[int] = None  # None => same as n_head (MHA). <n_head => grouped-query attention
    n_embd: int = 256              # model width
    mlp_ratio: float = 2.6667      # SwiGLU hidden = mlp_ratio * n_embd (8/3 keeps FLOPs ~= standard 4x FFN)
    mlp_multiple_of: int = 64      # round SwiGLU hidden up to a multiple of this (kernel-friendly)
    dropout: float = 0.0
    rope_base: float = 10000.0
    tie_word_embeddings: bool = True
    grad_checkpoint: bool = False  # trade ~25% speed for a big activation-memory cut

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        self.n_kv_head = self.n_kv_head or self.n_head
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})")

    # -- derived quantities ------------------------------------------------
    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def mlp_hidden(self) -> int:
        h = int(self.mlp_ratio * self.n_embd)
        h = max(self.mlp_multiple_of, math.ceil(h / self.mlp_multiple_of) * self.mlp_multiple_of)
        return h

    def param_count(self) -> int:
        """Analytic parameter count — no need to build the module to size a run."""
        e, h, m, L = self.n_embd, self.head_dim, self.mlp_hidden, self.n_layer
        d = self.head_dim
        embed = self.vocab_size * e
        attn = L * (e * (self.n_head * d) + 2 * e * (self.n_kv_head * d) + (self.n_head * d) * e)
        mlp = L * 3 * e * m
        norms = L * 2 * e + e  # 2 RMSNorms per block + final norm
        head = 0 if self.tie_word_embeddings else self.vocab_size * e
        return embed + attn + mlp + norms + head

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class TrainConfig:
    """Optimiser + schedule + loop settings."""

    out_dir: str = "runs/default"
    data_dir: str = "data"
    device: str = "auto"           # auto | cuda | cpu | mps
    dtype: str = "auto"            # auto | bf16 | fp16 | fp32
    batch_size: int = 16           # sequences per micro-step
    grad_accum: int = 1            # micro-steps per optimiser step
    max_steps: int = 2000
    learning_rate: float = 6e-4
    min_lr_frac: float = 0.1       # cosine floor as a fraction of learning_rate
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 100
    lr_decay: str = "cosine"       # cosine | linear | constant
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 10
    save_interval: int = 500
    keep_best: bool = True
    seed: int = 1337
    compile: bool = False          # torch.compile — faster, slower to start

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Presets. `params` is approximate; check ModelConfig.param_count() for exact.
# ---------------------------------------------------------------------------
PRESETS: dict[str, ModelConfig] = {
    # Tiny sanity model: trains in minutes on a CPU. Used by the test suite and
    # by `make smoke` to prove the whole pipeline works end to end.
    "smoke": ModelConfig(vocab_size=4096, block_size=256, n_layer=2, n_head=2, n_embd=64),

    # Fits in 4 GB VRAM (RTX 3050 laptop) with batch_size 8 + grad_accum 8.
    # ~13M params. Learns real structure on a narrow corpus.
    "rtx3050_4gb": ModelConfig(vocab_size=8192, block_size=512, n_layer=6, n_head=6, n_embd=384),

    # 8 GB VRAM (RTX 3050 desktop / 3060): ~30M params, 1024-token context.
    "rtx3050_8gb": ModelConfig(vocab_size=8192, block_size=1024, n_layer=8, n_head=8, n_embd=512),

    # 12-16 GB (3060 12GB / 4070): ~60M params.
    "small": ModelConfig(vocab_size=16384, block_size=1024, n_layer=10, n_head=10, n_embd=640),

    # 24 GB (3090 / 4090): ~100M params, GQA to shrink the KV cache.
    "medium": ModelConfig(
        vocab_size=32768, block_size=2048, n_layer=12, n_head=12, n_kv_head=4, n_embd=768
    ),
}


def pick_preset_for_vram(vram_gb: Optional[float] = None, device: str = "auto") -> str:
    """Choose the largest preset that comfortably fits the available VRAM.

    Falls back to CPU-friendly presets when there is no CUDA device.
    """
    if vram_gb is None:
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        else:
            return "smoke"

    if vram_gb >= 20:
        return "medium"
    if vram_gb >= 11:
        return "small"
    if vram_gb >= 7:
        return "rtx3050_8gb"
    if vram_gb >= 3.5:
        return "rtx3050_4gb"
    return "smoke"


def describe_presets() -> str:
    lines = []
    for name, cfg in PRESETS.items():
        lines.append(
            f"{name:<14} {cfg.param_count() / 1e6:7.1f}M params  "
            f"L={cfg.n_layer} W={cfg.n_embd} H={cfg.n_head} KV={cfg.n_kv_head} ctx={cfg.block_size}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_presets())
