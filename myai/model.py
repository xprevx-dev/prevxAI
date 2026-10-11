"""Decoder-only transformer, written from scratch.

Design choices (all standard for 2024-era small LLMs, all implemented here):
  * RMSNorm (pre-norm) instead of LayerNorm — one less statistic, same quality
  * Rotary position embeddings (RoPE) instead of learned absolute positions
  * SwiGLU MLP instead of GELU FFN
  * Grouped-query attention: n_kv_head may be < n_head to shrink the KV cache
  * Attention runs through F.scaled_dot_product_attention, which dispatches to
    the FlashAttention kernel when the GPU supports it
  * KV cache for token-at-a-time generation
  * Optional activation checkpointing to trade speed for VRAM
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from myai.config import ModelConfig


class RMSNorm(nn.Module):
    """Root-mean-square layer norm: x / sqrt(mean(x^2) + eps) * gain."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # upcast to fp32 for the reduction, cast back — this is what llama.cpp does
        # and it keeps bf16 training stable.
        dtype = x.dtype
        xf = x.float()
        norm = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight.float()).to(dtype)


def _build_rope_cache(head_dim: int, max_len: int, base: float, device, dtype):
    """Return (cos, sin) each of shape (max_len, head_dim)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                     # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)              # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D); cos/sin: (B, T, D)."""
    return (x * cos) + (_rotate_half(x) * sin)


class KVCache:
    """Growable per-layer key/value cache for autoregressive decoding."""

    def __init__(self, n_layer: int, n_kv_head: int, head_dim: int, device, dtype):
        self.k: List[Optional[torch.Tensor]] = [None] * n_layer
        self.v: List[Optional[torch.Tensor]] = [None] * n_layer
        self.n_layer = n_layer
        self.device = device
        self.dtype = dtype
        self._shape_tail = (n_kv_head, head_dim)

    @property
    def length(self) -> int:
        return 0 if self.k[0] is None else self.k[0].shape[2]

    def past_len(self, layer: int) -> int:
        """Cached length *for this layer*.

        Must be per-layer: within one forward pass, layer 0 is updated before
        layer 1 runs, so reading layer 0's length would make every later layer
        think it already has a cache and build a wrong attention mask.
        """
        k = self.k[layer]
        return 0 if k is None else k.shape[2]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v], dim=2)
        return self.k[layer], self.v[layer]  # type: ignore[return-value]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head or cfg.n_head
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_head // self.n_kv_head
        self.dropout = cfg.dropout

        self.q_proj = nn.Linear(cfg.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, cfg.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: Optional[KVCache] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        past_len = cache.past_len(layer_idx) if cache is not None else 0
        if cache is not None:
            k, v = cache.update(layer_idx, k, v)

        if self.n_rep > 1:  # grouped-query attention: expand KV heads to match Q heads
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # is_causal is only valid when the query length equals the key length,
        # i.e. on prefill with an empty cache. Decoding a single new token needs
        # no mask at all; decoding several tokens onto a non-empty cache needs an
        # explicitly offset causal mask.
        if past_len == 0:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=T > 1, dropout_p=self.dropout if self.training else 0.0
            )
        elif T == 1:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            i = torch.arange(T, device=x.device)[:, None]
            j = torch.arange(past_len + T, device=x.device)[None, :]
            mask = (j > i + past_len).to(q.dtype) * float("-inf")
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None], dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.resid_dropout(self.o_proj(y))


class MLP(nn.Module):
    """SwiGLU: down_proj( silu(gate_proj(x)) * up_proj(x) )."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        h = cfg.mlp_hidden
        self.gate_proj = nn.Linear(cfg.n_embd, h, bias=False)
        self.up_proj = nn.Linear(cfg.n_embd, h, bias=False)
        self.down_proj = nn.Linear(h, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cache=None, layer_idx=0):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache, layer_idx)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class MyGPT(nn.Module):
    """The language model. Predicts the next token from the previous ones."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # rope cache is a buffer: moves to device/dtype with the model, not trained
        cos, sin = _build_rope_cache(cfg.head_dim, cfg.block_size, cfg.rope_base, "cpu", torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scaled init for residual projections keeps early-training activations tame
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def configure_optimizers(self, weight_decay: float, betas: Tuple[float, float], device_type: str):
        """AdamW with weight decay applied only to 2-D matrices (never norms/biases)."""
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=1e-3, betas=betas, fused=fused)

    def _rope_for(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.rope_cos.index_select(0, positions)[None, :, :]  # (1, T, D)
        sin = self.rope_sin.index_select(0, positions)[None, :, :]
        return cos, sin

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        past_len = cache.length if cache is not None else 0
        if positions is None:
            positions = torch.arange(past_len, past_len + T, device=idx.device)
        cos, sin = self._rope_for(positions)

        x = self.tok_emb(idx)
        for i, block in enumerate(self.blocks):
            if self.cfg.grad_checkpoint and self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(
                    block, x, cos, sin, cache, i, use_reentrant=False
                )
            else:
                x = block(x, cos, sin, cache, i)
        x = self.norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
            return logits, loss

        # generation path: we only need logits for the final position
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    # -- generation --------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Sample up to `max_new_tokens` continuations of `idx`.

        Returns prompt + generated tokens, shape (B, T_prompt + n_new).

        RoPE positions only exist up to block_size - 1, so once the running
        sequence fills the context the window slides: the oldest tokens are
        dropped and the cache is rebuilt. `all_ids` still holds everything that
        was produced, so the return value is never truncated.
        """
        self.eval()
        B = idx.size(0)
        device = idx.device
        limit = self.cfg.block_size
        keep = max(1, limit - 1)

        def new_cache() -> Optional[KVCache]:
            if not use_cache:
                return None
            return KVCache(self.cfg.n_layer, self.cfg.n_kv_head or self.cfg.n_head,
                           self.cfg.head_dim, device, self.tok_emb.weight.dtype)

        # a prompt longer than the context is truncated to its tail
        all_ids = idx[:, -limit:] if idx.size(1) > limit else idx
        window = all_ids
        cache = new_cache()

        logits, _ = self(window, cache=cache)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            step_logits = logits[:, -1, :] / max(temperature, 1e-5)

            if repetition_penalty != 1.0:
                seen = torch.zeros_like(step_logits, dtype=torch.bool)
                rows = torch.arange(B, device=device).repeat_interleave(all_ids.size(1))
                seen[rows, all_ids.reshape(-1)] = True
                # HF-style: shrink positive logits, deepen negative ones — both
                # directions lower the repeated token's probability.
                if repetition_penalty > 1.0:
                    step_logits = torch.where(
                        seen & (step_logits > 0), step_logits / repetition_penalty, step_logits)
                    step_logits = torch.where(
                        seen & (step_logits <= 0), step_logits * repetition_penalty, step_logits)
                else:
                    step_logits = torch.where(
                        seen & (step_logits > 0), step_logits * repetition_penalty, step_logits)
                    step_logits = torch.where(
                        seen & (step_logits <= 0), step_logits / repetition_penalty, step_logits)

            if top_k is not None:
                k = min(top_k, step_logits.size(-1))
                v, _ = torch.topk(step_logits, k)
                step_logits = step_logits.masked_fill(step_logits < v[:, [-1]], float("-inf"))

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(step_logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum = probs.cumsum(dim=-1)
                remove = cum > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                step_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(step_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if eos_id is not None:
                # frozen sequences emit EOS so every row stays the same length
                next_id = torch.where(finished[:, None], torch.full_like(next_id, eos_id), next_id)
            all_ids = torch.cat([all_ids, next_id], dim=1)
            if eos_id is not None:
                finished = finished | (next_id[:, 0] == eos_id)
                if bool(finished.all()):
                    break

            if all_ids.size(1) > keep:
                # context full: slide the window and start the cache over
                window = all_ids[:, -keep:]
                cache = new_cache()
                logits, _ = self(window, cache=cache)
            else:
                window = all_ids if not use_cache else next_id
                logits, _ = self(window, cache=cache)

        return all_ids

def load_checkpoint_model(path: str, device: str = "cpu", dtype=None):
    """Rebuild a model + tokenizer + configs from a training checkpoint."""
    from myai.tokenizer import BPETokenizer

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["model_config"])
    model = MyGPT(cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    if dtype is not None:
        model = model.to(dtype)
    model.eval()
    tok = BPETokenizer.from_dict(ckpt["tokenizer"])
    return model, tok, ckpt.get("train_config"), ckpt
