"""Model correctness: shapes, causality, gradients, RoPE, and the KV cache."""

import math

import pytest
import torch

from myai.config import PRESETS, ModelConfig
from myai.model import KVCache, MyGPT, RMSNorm, apply_rope


def small_cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=128, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    base.update(kw)
    return ModelConfig(**base)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = MyGPT(small_cfg())
    m.eval()
    return m


def test_forward_shapes(model):
    x = torch.randint(0, 128, (3, 17))
    logits, loss = model(x, x)
    assert logits.shape == (3, 17, 128)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_inference_returns_last_position_only(model):
    x = torch.randint(0, 128, (2, 9))
    logits, loss = model(x)
    assert logits.shape == (2, 1, 128)
    assert loss is None


def test_initial_loss_near_uniform(model):
    """A fresh model should sit near ln(vocab) — a sanity check on the init.

    ln(128) = 4.85. The scaled residual init pulls it a little below uniform,
    so allow a wide band: the point is to catch an init that is wildly off.
    """
    x = torch.randint(0, 128, (8, 32))
    _, loss = model(x, x)
    assert abs(loss.item() - math.log(128)) < 1.0


def test_causality(model):
    """Position t must not be able to see tokens after t."""
    x = torch.randint(0, 128, (1, 24))
    logits, _ = model(x, x)

    x2 = x.clone()
    x2[0, 10:] = torch.randint(0, 128, (14,))
    logits2, _ = model(x2, x2)

    assert torch.allclose(logits[0, :10], logits2[0, :10], atol=1e-5), \
        "future tokens leaked into earlier positions"
    assert not torch.allclose(logits[0, 10:], logits2[0, 10:], atol=1e-5), \
        "changed tokens should change later logits"


def test_gradients_reach_every_parameter():
    torch.manual_seed(1)
    m = MyGPT(small_cfg())
    x = torch.randint(0, 128, (2, 16))
    _, loss = m(x, x)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
        assert p.grad.abs().sum() > 0, f"zero grad for {name}"


def test_param_count_matches_analytic():
    for name, cfg in PRESETS.items():
        m = MyGPT(cfg)
        actual = m.num_params()
        assert actual == cfg.param_count(), (
            f"{name}: analytic {cfg.param_count()} != actual {actual}"
        )


def test_kv_cache_matches_full_recompute(model):
    """Incremental decoding must produce the same logits as one full pass."""
    x = torch.randint(0, 128, (1, 20))
    full, _ = model(x, x)

    cache = KVCache(2, 2, 16, torch.device("cpu"), torch.float32)
    with torch.no_grad():
        prefill, _ = model(x[:, :6], cache=cache)
        chunks = [prefill[:, -1:, :]]
        for i in range(6, 20):
            step_logits, _ = model(x[:, i : i + 1], cache=cache)
            chunks.append(step_logits[:, -1:, :])
        inc = torch.cat(chunks, dim=1)

    assert cache.length == 20
    assert inc.shape == full[:, 5:, :].shape
    assert torch.allclose(full[:, 5:, :], inc, atol=2e-4), \
        f"max diff {(full[:, 5:, :] - inc).abs().max().item()}"


def test_cacheless_generation_matches_cached():
    torch.manual_seed(3)
    m = MyGPT(small_cfg())
    m.eval()
    x = torch.randint(0, 128, (1, 12))
    torch.manual_seed(99)
    a = m.generate(x, max_new_tokens=6, temperature=1.0, use_cache=True)
    torch.manual_seed(99)
    b = m.generate(x, max_new_tokens=6, temperature=1.0, use_cache=False)
    assert a.shape == (1, 18)
    assert torch.equal(a, b), "KV cache changed the sampled output"


def test_generate_beyond_context_window(model):
    """Generation must keep working past block_size by sliding the window."""
    x = torch.randint(0, 128, (1, 32))          # prompt already fills the context
    out = model.generate(x, max_new_tokens=10, temperature=1.0)
    assert out.shape == (1, 42)                  # nothing is dropped from the output
    assert torch.equal(out[0, :32], x[0])        # the prompt is preserved verbatim

    long_prompt = torch.randint(0, 128, (1, 50))  # longer than the context
    out2 = model.generate(long_prompt, max_new_tokens=4, temperature=1.0)
    assert out2.shape == (1, 36)                  # truncated to the tail, then extended
    assert torch.equal(out2[0, :32], long_prompt[0, -32:])


def test_grouped_query_attention_runs():
    cfg = small_cfg(n_head=4, n_embd=32, n_kv_head=2)
    m = MyGPT(cfg)
    m.eval()
    x = torch.randint(0, 128, (2, 11))
    logits, loss = m(x, x)
    assert logits.shape == (2, 11, 128)
    # fewer KV heads than Q heads must actually shrink those weight matrices
    assert m.blocks[0].attn.k_proj.weight.shape[0] == 2 * cfg.head_dim
    assert m.blocks[0].attn.q_proj.weight.shape[0] == 4 * cfg.head_dim


def test_invalid_head_config_rejected():
    with pytest.raises(ValueError):
        ModelConfig(n_embd=32, n_head=3)
    with pytest.raises(ValueError):
        ModelConfig(n_embd=32, n_head=4, n_kv_head=3)


def test_rmsnorm_normalises():
    n = RMSNorm(16)
    x = torch.randn(4, 16) * 37
    y = n(x)
    assert y.shape == x.shape
    rms = y.float().pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_rope_is_position_dependent_and_stable():
    from myai.model import _build_rope_cache

    cos, sin = _build_rope_cache(16, 64, 10000.0, torch.device("cpu"), torch.float32)
    assert cos.shape == (64, 16) and sin.shape == (64, 16)
    # position 0 must be the identity rotation
    assert torch.allclose(cos[0], torch.ones(16), atol=1e-6)
    assert torch.allclose(sin[0], torch.zeros(16), atol=1e-6)
    q = torch.randn(1, 2, 64, 16)
    rotated = apply_rope(q, cos[None, :, :], sin[None, :, :])
    # rotation preserves vector norm
    assert torch.allclose(
        rotated.norm(dim=-1), q.norm(dim=-1), atol=1e-5
    )


def test_tied_embeddings():
    m = MyGPT(small_cfg())
    assert m.lm_head.weight is m.tok_emb.weight
    m2 = MyGPT(small_cfg(tie_word_embeddings=False))
    assert m2.lm_head.weight is not m2.tok_emb.weight
    assert m2.num_params() > m.num_params()


def test_grad_checkpoint_matches_eager():
    torch.manual_seed(7)
    cfg = small_cfg()
    a = MyGPT(cfg)
    cfg_gc = small_cfg(grad_checkpoint=True)
    b = MyGPT(cfg_gc)
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 128, (2, 16))
    a.train(); b.train()
    _, la = a(x, x)
    _, lb = b(x, x)
    assert torch.allclose(la, lb, atol=1e-5)
    lb.backward()
    assert b.blocks[0].attn.q_proj.weight.grad is not None
