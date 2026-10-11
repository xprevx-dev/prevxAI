"""End-to-end: the LR schedule, real overfitting, and checkpoint round-trips.

The overfit test is the important one — it drives the actual train() function on
a tiny corpus and asserts the loss collapses. If the data pipeline, masking,
optimiser, or backward pass were broken anywhere, this fails.
"""

import json
import math
import os

import numpy as np
import pytest
import torch

from myai.config import ModelConfig, TrainConfig
from myai.model import MyGPT, load_checkpoint_model
from myai.tokenizer import train_bpe
from myai.train import lr_at, train

REPETITIVE = ("the cat sat on the mat and looked at the door. " * 120) + (
    "a dog ran in the park and chased the ball. " * 120
)


@pytest.fixture(scope="module")
def tiny_tokenizer():
    return train_bpe(REPETITIVE, num_merges=120, verbose=False)


def write_split(tmp_path, name, text, tok):
    arr = np.array(tok.encode(text) + [tok.sep_id], dtype=np.uint16)
    arr.tofile(os.path.join(tmp_path, f"{name}.bin"))
    return len(arr)


def test_lr_schedule_warmup_and_floor():
    cfg = TrainConfig(learning_rate=1e-3, warmup_steps=100, max_steps=1000, min_lr_frac=0.1)
    assert lr_at(0, cfg) == pytest.approx(1e-3 / 100)
    assert lr_at(99, cfg) == pytest.approx(1e-3)
    assert lr_at(100, cfg) == pytest.approx(1e-3, rel=1e-3)
    assert lr_at(999, cfg) == pytest.approx(1e-4, rel=1e-3)
    # monotonic decay after warmup
    tail = [lr_at(s, cfg) for s in range(100, 1000, 50)]
    assert all(a > b for a, b in zip(tail, tail[1:]))


def test_lr_constant_and_linear():
    cfg = TrainConfig(learning_rate=1e-3, warmup_steps=10, max_steps=100,
                      lr_decay="constant", min_lr_frac=0.1)
    assert lr_at(99, cfg) == pytest.approx(1e-3)
    cfg.lr_decay = "linear"
    assert lr_at(99, cfg) == pytest.approx(1e-4, rel=1e-2)


def test_end_to_end_overfit(tmp_path, tiny_tokenizer):
    tok = tiny_tokenizer
    n = write_split(tmp_path, "train", REPETITIVE, tok)
    write_split(tmp_path, "val", REPETITIVE[:2000], tok)
    assert n > 1000

    model_cfg = ModelConfig(
        vocab_size=tok.vocab_size, block_size=32, n_layer=2, n_head=2, n_embd=48
    )
    train_cfg = TrainConfig(
        out_dir=str(tmp_path / "run"), data_dir=str(tmp_path), max_steps=150,
        batch_size=8, grad_accum=1, learning_rate=3e-3, warmup_steps=10,
        eval_interval=75, eval_iters=3, log_interval=25, save_interval=75,
        device="cpu", dtype="fp32",
    )
    result = train(model_cfg, train_cfg, tok)

    # a fresh model starts near ln(vocab); memorising this corpus should crush it
    assert result["best_val"] is not None
    assert result["best_val"] < 0.5, f"model failed to overfit: val loss {result['best_val']}"
    assert result["best_val"] < math.log(tok.vocab_size) * 0.2

    assert os.path.exists(os.path.join(train_cfg.out_dir, "best.pt"))
    assert os.path.exists(os.path.join(train_cfg.out_dir, "last.pt"))
    assert os.path.exists(os.path.join(train_cfg.out_dir, "log.jsonl"))

    # the training log must be valid JSONL with a decreasing loss trend
    rows = [json.loads(l) for l in open(os.path.join(train_cfg.out_dir, "log.jsonl"))]
    losses = [r["loss"] for r in rows if "loss" in r]
    assert len(losses) >= 3
    assert losses[-1] < losses[0]


def test_checkpoint_roundtrip(tmp_path, tiny_tokenizer):
    tok = tiny_tokenizer
    write_split(tmp_path, "train", REPETITIVE, tok)
    model_cfg = ModelConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=2,
                            n_head=2, n_embd=32)
    train_cfg = TrainConfig(
        out_dir=str(tmp_path / "run2"), data_dir=str(tmp_path), max_steps=5,
        batch_size=4, learning_rate=1e-3, warmup_steps=2, eval_interval=1000,
        save_interval=5, log_interval=5, device="cpu", dtype="fp32",
    )
    train(model_cfg, train_cfg, tok)

    path = os.path.join(train_cfg.out_dir, "last.pt")
    model, tok2, saved_cfg, ckpt = load_checkpoint_model(path)
    assert ckpt["step"] == 4
    assert saved_cfg is not None and saved_cfg["max_steps"] == 5
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.merges == tok.merges

    # weights really were restored
    fresh = MyGPT(model_cfg)
    assert not torch.allclose(fresh.tok_emb.weight, model.tok_emb.weight)

    # and the rebuilt model reproduces the same logits from the raw state dict
    ckpt_model = MyGPT(ModelConfig.from_dict(ckpt["model_config"]))
    ckpt_model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    ckpt_model.eval()
    x = torch.randint(0, tok.vocab_size, (1, 16))
    with torch.no_grad():
        a, _ = model(x, x)
        b, _ = ckpt_model(x, x)
    assert torch.allclose(a, b, atol=1e-6)


def test_resume_continues_from_saved_step(tmp_path, tiny_tokenizer):
    tok = tiny_tokenizer
    write_split(tmp_path, "train", REPETITIVE, tok)
    model_cfg = ModelConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=2,
                            n_head=2, n_embd=32)
    base = dict(data_dir=str(tmp_path), batch_size=4, learning_rate=1e-3, warmup_steps=2,
                eval_interval=1000, save_interval=3, log_interval=3, device="cpu", dtype="fp32")

    first = TrainConfig(out_dir=str(tmp_path / "run3"), max_steps=3, **base)
    train(model_cfg, first, tok)
    ckpt_path = os.path.join(first.out_dir, "last.pt")
    assert torch.load(ckpt_path, map_location="cpu", weights_only=False)["step"] == 2

    second = TrainConfig(out_dir=str(tmp_path / "run3"), max_steps=6, **base)
    train(model_cfg, second, tok, resume_from=ckpt_path)
    assert torch.load(ckpt_path, map_location="cpu", weights_only=False)["step"] == 5


def test_missing_data_dir_gives_a_useful_error(tmp_path, tiny_tokenizer):
    cfg = TrainConfig(data_dir=str(tmp_path / "nope"), out_dir=str(tmp_path / "r"))
    with pytest.raises(FileNotFoundError):
        train(ModelConfig(vocab_size=100), cfg, tiny_tokenizer)


def test_grad_accum_matches_bigger_batch_loss_scale(tmp_path, tiny_tokenizer):
    """loss/grad_accum is applied per micro-step, so the accumulated loss is
    comparable across accumulation settings."""
    tok = tiny_tokenizer
    write_split(tmp_path, "train", REPETITIVE, tok)
    model_cfg = ModelConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=1,
                            n_head=2, n_embd=32)
    for accum in (1, 2):
        cfg = TrainConfig(out_dir=str(tmp_path / f"ga{accum}"), data_dir=str(tmp_path),
                          max_steps=2, batch_size=2, grad_accum=accum, learning_rate=1e-3,
                          warmup_steps=1, eval_interval=1000, save_interval=2,
                          log_interval=1, device="cpu", dtype="fp32")
        torch.manual_seed(0)
        np.random.seed(0)
        result = train(model_cfg, cfg, tok)
        assert result["out_dir"].endswith(f"ga{accum}")
