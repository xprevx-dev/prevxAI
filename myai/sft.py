"""Supervised fine-tuning: turn the base model into a chatbot.

Pre-training teaches the model English (or whatever corpus you used). SFT teaches
it the *shape* of answering: read the user turn, reply, emit <|eot|>.

Examples are packed into blocks and loss is computed only on assistant tokens,
so the model never learns to imitate the questions.

    python -m myai.sft --checkpoint runs/chatbot/best.pt --data data/chat.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from myai.chat import DEFAULT_SYSTEM, IGNORE_INDEX, Message, build_training_example, tokenize_with_mask
from myai.config import ModelConfig, TrainConfig
from myai.model import MyGPT
from myai.tokenizer import BPETokenizer
from myai.train import lr_at, resolve_device, resolve_dtype, save_checkpoint


def build_packed_arrays(
    convos: Sequence[Sequence[Message]],
    tokenizer: BPETokenizer,
    block_size: int,
    system: Optional[str] = DEFAULT_SYSTEM,
    drop_incomplete: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tokenise + label every conversation, then concatenate into one stream.

    Returns (ids, labels) as int32 arrays. A block boundary can fall in the middle
    of a conversation; that is fine because labels carry their own mask.
    """
    ids: List[int] = []
    labels: List[int] = []
    skipped = 0
    for convo in convos:
        ex = build_training_example(convo, system=system)
        tok_ids, tok_labels = tokenize_with_mask(tokenizer, ex.text, ex.assistant_spans)
        if not tok_ids:
            skipped += 1
            continue
        if drop_incomplete and len(tok_ids) > block_size:
            skipped += 1
            continue
        ids.extend(tok_ids)
        labels.extend(tok_labels)
        ids.append(tokenizer.eos_id)
        labels.append(IGNORE_INDEX)
    if skipped:
        print(f"[sft] skipped {skipped} conversations (empty or longer than block_size)")
    return np.array(ids, dtype=np.int32), np.array(labels, dtype=np.int32)


def random_batch(ids: np.ndarray, labels: np.ndarray, block_size: int, batch_size: int, device: str):
    hi = len(ids) - block_size - 1
    if hi < 1:
        raise ValueError(
            f"need more than block_size tokens to form a batch: got {len(ids)}, "
            f"block_size={block_size}"
        )
    ix = np.random.randint(0, hi, size=(batch_size,))
    x = np.stack([ids[i : i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([labels[i + 1 : i + 1 + block_size] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def sft(
    model: MyGPT,
    tokenizer: BPETokenizer,
    ids: np.ndarray,
    labels: np.ndarray,
    val_ids: Optional[np.ndarray],
    val_labels: Optional[np.ndarray],
    cfg: TrainConfig,
    model_cfg: ModelConfig,
) -> dict:
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype, device)
    use_amp = device == "cuda" and dtype in (torch.float16, torch.bfloat16)
    ctx = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if use_amp
        else torch.autocast(device_type="cuda", enabled=False)
    )

    model = model.to(device)
    optimizer = model.configure_optimizers(cfg.weight_decay, (cfg.beta1, cfg.beta2),
                                          "cuda" if device == "cuda" else "cpu")
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    n_masked = int((labels != IGNORE_INDEX).sum())
    print(f"[sft] {len(ids):,} packed tokens, {n_masked:,} of them supervised")
    if n_masked == 0:
        raise SystemExit("[sft] no supervised tokens — check that your data has assistant turns")

    best_val: Optional[float] = None
    model.train()
    for step in range(cfg.max_steps):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step, cfg)
        optimizer.zero_grad(set_to_none=True)
        accum = 0.0
        for _ in range(cfg.grad_accum):
            X, Y = random_batch(ids, labels, model_cfg.block_size, cfg.batch_size, device)
            with ctx:
                _, loss = model(X, Y)
                loss = loss / cfg.grad_accum
            scaler.scale(loss).backward()
            accum += loss.item()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if (step + 1) % cfg.log_interval == 0:
            print(f"[sft] step {step + 1}/{cfg.max_steps} | loss {accum:.4f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}")

        if val_ids is not None and len(val_ids) > model_cfg.block_size + 1 \
                and (step + 1) % cfg.eval_interval == 0:
            model.eval()
            losses = []
            with torch.no_grad():
                for _ in range(cfg.eval_iters):
                    X, Y = random_batch(val_ids, val_labels, model_cfg.block_size,
                                        cfg.batch_size, device)
                    with ctx:
                        _, l = model(X, Y)
                    losses.append(l.item())
            val = float(np.mean(losses))
            print(f"[sft] eval step {step + 1} | val loss {val:.4f}")
            model.train()
            if cfg.keep_best and (best_val is None or val < best_val):
                best_val = val
                save_checkpoint(os.path.join(cfg.out_dir, "sft_best.pt"), model, optimizer,
                                scaler, step, best_val, model_cfg, cfg, tokenizer)

    save_checkpoint(os.path.join(cfg.out_dir, "sft.pt"), model, optimizer, scaler,
                    cfg.max_steps - 1, best_val, model_cfg, cfg, tokenizer)
    print(f"[sft] saved {os.path.join(cfg.out_dir, 'sft.pt')}")
    return {"best_val": best_val}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Instruction fine-tune a checkpoint.")
    p.add_argument("--checkpoint", required=True, help="Pretrained .pt to start from")
    p.add_argument("--data", required=True, help=".jsonl of conversations")
    p.add_argument("--out-dir", default=None, help="Default: alongside the checkpoint")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--learning-rate", type=float, default=1e-4,
                   help="Lower than pretraining — we are nudging, not retraining")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="0 for SFT: we are nudging a pretrained model, not regularising from scratch")
    p.add_argument("--system", default=DEFAULT_SYSTEM)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args(argv)

    from myai.model import load_checkpoint_model

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model, tokenizer, saved_train_cfg, ckpt = load_checkpoint_model(args.checkpoint)
    model_cfg = ModelConfig.from_dict(ckpt["model_config"])
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.checkpoint))

    from myai.chat import messages_from_jsonl

    convos = messages_from_jsonl(args.data)
    random.shuffle(convos)
    n_val = max(1, int(len(convos) * args.val_frac)) if len(convos) > 4 else 0
    val_convos, train_convos = convos[:n_val], convos[n_val:]
    print(f"[sft] {len(train_convos)} train / {len(val_convos)} val conversations")

    ids, labels = build_packed_arrays(train_convos, tokenizer, model_cfg.block_size, args.system)
    val_ids = val_labels = None
    if val_convos:
        val_ids, val_labels = build_packed_arrays(val_convos, tokenizer,
                                                  model_cfg.block_size, args.system)
        if len(val_ids) <= model_cfg.block_size + 1:
            # a val split shorter than one window cannot produce a batch at all;
            # better to train without validation than to crash halfway through
            print(f"[sft] val split too small ({len(val_ids)} tokens < block_size "
                  f"{model_cfg.block_size}); training without validation. "
                  "Add more conversations or lower the model's block_size.")
            val_ids = val_labels = None

    cfg = TrainConfig(
        out_dir=out_dir, max_steps=args.max_steps, learning_rate=args.learning_rate,
        batch_size=args.batch_size, grad_accum=args.grad_accum, device=args.device,
        dtype=args.dtype, seed=args.seed, warmup_steps=max(10, args.max_steps // 10),
        eval_interval=max(50, args.max_steps // 6), log_interval=10,
        save_interval=max(50, args.max_steps // 4),
        weight_decay=args.weight_decay,
    )
    if len(ids) <= model_cfg.block_size + 2:
        raise SystemExit(
            f"[sft] only {len(ids)} tokens of training data — add more conversations "
            "or lower the model's block_size."
        )
    sft(model, tokenizer, ids, labels, val_ids, val_labels, cfg, model_cfg)


if __name__ == "__main__":
    main()
