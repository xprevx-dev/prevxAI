"""Training loop: AdamW, warmup + cosine decay, bf16, gradient accumulation,
checkpointing with resume, and periodic validation.

Run it as a script:

    python -m myai.train --preset rtx3050_4gb --data-dir data --out-dir runs/chatbot
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch

from myai.config import PRESETS, ModelConfig, TrainConfig, describe_presets
from myai.data import load_bin, random_batch, read_meta
from myai.model import MyGPT
from myai.tokenizer import BPETokenizer


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    if requested == "fp32":
        return torch.float32
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    # auto: bf16 on anything that supports it (Ampere and newer — the 3050 does)
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "mps":
        return torch.bfloat16
    return torch.float32


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then decay to min_lr_frac * learning_rate."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    floor = cfg.min_lr_frac * cfg.learning_rate
    if cfg.lr_decay == "constant":
        return cfg.learning_rate
    # -1 so the final step lands exactly on the floor
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - 1 - cfg.warmup_steps)
    progress = min(1.0, progress)
    if cfg.lr_decay == "linear":
        return cfg.learning_rate + (floor - cfg.learning_rate) * progress
    return floor + 0.5 * (cfg.learning_rate - floor) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(
    model: MyGPT,
    splits: dict,
    cfg: TrainConfig,
    device: str,
    ctx,
) -> dict:
    """Average loss over `eval_iters` random windows of each split."""
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            X, Y = random_batch(data, model.cfg.block_size, cfg.batch_size, device)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(
    path: str,
    model: MyGPT,
    optimizer,
    scaler,
    step: int,
    best_val: Optional[float],
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    tokenizer: BPETokenizer,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "best_val": best_val,
            "model_config": model_cfg.to_dict(),
            "train_config": train_cfg.to_dict(),
            "tokenizer": tokenizer.to_dict(),
        },
        path,
    )


def load_data_splits(data_dir: str) -> Tuple[np.memmap, Optional[np.memmap]]:
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"{train_path} not found. Run: python scripts/prepare_data.py --input <corpus>"
        )
    train = load_bin(train_path)
    val = load_bin(val_path) if os.path.exists(val_path) else None
    return train, val


def train(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    tokenizer: BPETokenizer,
    resume_from: Optional[str] = None,
) -> dict:
    device = resolve_device(train_cfg.device)
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(train_cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dtype = resolve_dtype(train_cfg.dtype, device)
    device_type = "cuda" if device == "cuda" else "cpu"
    # mixed precision only pays off on GPU; on CPU we just run fp32
    use_amp = device == "cuda" and dtype in (torch.float16, torch.bfloat16)
    ctx = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if use_amp
        else torch.autocast(device_type="cuda", enabled=False)
    )

    train_data, val_data = load_data_splits(train_cfg.data_dir)
    meta = read_meta(train_cfg.data_dir)
    print(f"[train] device={device} dtype={dtype} amp={use_amp}")
    print(f"[train] train tokens={len(train_data):,} "
          f"val tokens={len(val_data) if val_data is not None else 0:,}")
    print(f"[train] model: {model_cfg.param_count() / 1e6:.1f}M params, "
          f"context {model_cfg.block_size}, vocab {model_cfg.vocab_size}")
    if meta:
        print(f"[train] data meta: {meta}")

    tokens_per_step = train_cfg.batch_size * train_cfg.grad_accum * model_cfg.block_size

    model = MyGPT(model_cfg).to(device)
    optimizer = model.configure_optimizers(
        train_cfg.weight_decay, (train_cfg.beta1, train_cfg.beta2), device_type
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    start_step = 0
    best_val: Optional[float] = None
    if resume_from and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt.get("step", 0) + 1
        best_val = ckpt.get("best_val")
        print(f"[train] resumed from {resume_from} at step {start_step}")

    if train_cfg.compile:
        print("[train] torch.compile enabled (first step is slow)")
        model = torch.compile(model)
        raw_model = model._orig_mod  # type: ignore[attr-defined]
    else:
        raw_model = model

    os.makedirs(train_cfg.out_dir, exist_ok=True)
    log_path = os.path.join(train_cfg.out_dir, "log.jsonl")
    running = {"loss": 0.0, "n": 0}
    t0 = time.time()
    tokens_since_log = 0

    model.train()
    for step in range(start_step, train_cfg.max_steps):
        for lr_group in optimizer.param_groups:
            lr_group["lr"] = lr_at(step, train_cfg)

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for micro in range(train_cfg.grad_accum):
            X, Y = random_batch(train_data, model_cfg.block_size, train_cfg.batch_size, device)
            with ctx:
                _, loss = model(X, Y)
                loss = loss / train_cfg.grad_accum
            scaler.scale(loss).backward()
            accum_loss += loss.item()
        if train_cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running["loss"] += accum_loss
        running["n"] += 1
        tokens_since_log += tokens_per_step

        if (step + 1) % train_cfg.log_interval == 0:
            dt = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            mean_loss = running["loss"] / running["n"]
            tps = tokens_since_log / dt
            print(
                f"step {step + 1:>6}/{train_cfg.max_steps} | loss {mean_loss:.4f} | "
                f"lr {lr:.2e} | {tps:,.0f} tok/s | {dt:.0f}s"
            )
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step": step + 1, "loss": round(mean_loss, 4), "lr": lr,
                    "tok_per_s": round(tps, 1), "elapsed_s": round(dt, 1),
                }) + "\n")
            running = {"loss": 0.0, "n": 0}
            t0 = time.time()
            tokens_since_log = 0

        do_eval = val_data is not None and (
            (step + 1) % train_cfg.eval_interval == 0 or step + 1 == train_cfg.max_steps
        )
        do_save = (step + 1) % train_cfg.save_interval == 0 or step + 1 == train_cfg.max_steps

        if do_eval:
            splits = {"val": val_data}
            losses = estimate_loss(raw_model, splits, train_cfg, device, ctx)
            val = losses["val"]
            print(f"[eval] step {step + 1} | val loss {val:.4f} | ppl {math.exp(min(val, 20)):.2f}")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step + 1, "val_loss": round(val, 4)}) + "\n")
            if train_cfg.keep_best and (best_val is None or val < best_val):
                best_val = val
                save_checkpoint(
                    os.path.join(train_cfg.out_dir, "best.pt"), raw_model, optimizer, scaler,
                    step, best_val, model_cfg, train_cfg, tokenizer,
                )
                print(f"[eval] new best val loss {val:.4f} -> best.pt")

        if do_save:
            save_checkpoint(
                os.path.join(train_cfg.out_dir, "last.pt"), raw_model, optimizer, scaler,
                step, best_val, model_cfg, train_cfg, tokenizer,
            )
            print(f"[ckpt] saved {os.path.join(train_cfg.out_dir, 'last.pt')}")

    print("[train] done.")
    return {"best_val": best_val, "out_dir": train_cfg.out_dir}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the model from scratch.")
    p.add_argument("--preset", default=None, choices=list(PRESETS),
                   help="Named architecture preset (see --list-presets)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="runs/default")
    p.add_argument("--resume", default=None, help="Checkpoint to resume from")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--n-layer", type=int, default=None)
    p.add_argument("--n-embd", type=int, default=None)
    p.add_argument("--n-head", type=int, default=None)
    p.add_argument("--vocab-size", type=int, default=None,
                   help="Default: match the tokenizer found in the data dir")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--grad-checkpoint", action="store_true", help="Save VRAM, cost ~25% speed")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--eval-interval", type=int, default=None)
    p.add_argument("--eval-iters", type=int, default=None,
                   help="Batches to average for validation loss")
    p.add_argument("--save-interval", type=int, default=None)
    p.add_argument("--log-interval", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--list-presets", action="store_true")
    p.add_argument("--auto-preset", action="store_true",
                   help="Pick the largest preset that fits the detected VRAM")
    return p


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.list_presets:
        print(describe_presets())
        return

    if args.preset is None:
        from myai.config import pick_preset_for_vram

        args.preset = pick_preset_for_vram()
        print(f"[train] auto-selected preset: {args.preset}")

    model_cfg = ModelConfig.from_dict(PRESETS[args.preset].to_dict())
    train_cfg = TrainConfig(out_dir=args.out_dir, data_dir=args.data_dir)

    for key in ("max_steps", "batch_size", "grad_accum", "learning_rate", "eval_interval",
                "eval_iters", "save_interval", "log_interval", "seed"):
        val = getattr(args, key)
        if val is not None:
            setattr(train_cfg, key, val)
    for key in ("device", "dtype", "compile"):
        setattr(train_cfg, key, getattr(args, key))
    for key in ("block_size", "n_layer", "n_embd", "n_head", "vocab_size"):
        val = getattr(args, key)
        if val is not None:
            setattr(model_cfg, key, val)
    model_cfg.grad_checkpoint = args.grad_checkpoint
    model_cfg.__post_init__()

    # the tokenizer on disk decides the real vocab size — never let them drift
    tok_path = os.path.join(args.data_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        tokenizer = BPETokenizer.load(tok_path)
        if args.vocab_size is None:
            model_cfg.vocab_size = tokenizer.vocab_size
        elif tokenizer.vocab_size != model_cfg.vocab_size:
            raise SystemExit(
                f"--vocab-size {model_cfg.vocab_size} != tokenizer vocab "
                f"{tokenizer.vocab_size}. Drop --vocab-size to use the tokenizer."
            )
    else:
        raise SystemExit(
            f"{tok_path} not found. Train the tokenizer first: "
            "python scripts/prepare_data.py --input <your-corpus>"
        )

    train(model_cfg, train_cfg, tokenizer, resume_from=args.resume)


if __name__ == "__main__":
    main()
