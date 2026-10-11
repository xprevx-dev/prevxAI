# myai — your own language model, built from scratch

A decoder-only transformer LLM and chatbot with **no model APIs and no pretrained
weights**. No OpenAI, no Anthropic, no Hugging Face `transformers`, no downloaded
`.safetensors`. PyTorch provides autograd and CUDA; everything else — BPE
tokenizer, transformer, data pipeline, training loop, sampler, chat template,
instruction tuning, HTTP server — is code in this repo.

```
myai/
├── myai/
│   ├── config.py      model + training configs, VRAM-sized presets
│   ├── tokenizer.py   byte-level BPE, trained from scratch on your corpus
│   ├── model.py       the transformer: RoPE, RMSNorm, SwiGLU, GQA, KV cache
│   ├── data.py        text -> memmap token stream -> batches
│   ├── train.py       pretraining loop: AdamW, cosine LR, bf16, checkpointing
│   ├── sft.py         instruction tuning with assistant-only loss masking
│   ├── chat.py        the chat template (used identically in train + inference)
│   ├── generate.py    sampling, chat completion, terminal REPL
│   └── serve.py       web UI + OpenAI-compatible API, stdlib only
├── scripts/
│   ├── prepare_data.py    corpus -> tokenizer.json + train.bin/val.bin
│   ├── train_rtx3050.sh   auto-sized one-command run
│   └── push.sh            push this repo to GitHub
└── tests/             52 tests: tokenizer, model, training, chat, server
```

---

## Quick start

```bash
pip install torch numpy pytest        # CPU wheel is fine to start

# 1. Prove the whole pipeline works on a CPU in about a minute
make smoke

# 2. Tokenise a corpus
python scripts/prepare_data.py --input ~/my_notes --out-dir data

# 3. Pretrain
python -m myai.train --preset rtx3050_4gb --data-dir data --out-dir runs/chatbot

# 4. Turn it into a chatbot
python -m myai.sft --checkpoint runs/chatbot/best.pt --data data/chat.jsonl

# 5. Talk to it
python -m myai.generate --checkpoint runs/chatbot/sft.pt --chat
python -m myai.serve    --checkpoint runs/chatbot/sft.pt --port 8000
```

On your machine, `./scripts/train_rtx3050.sh` does steps 2-4 and sizes the model
to the VRAM it detects.

---

## The four stages

**1. Tokenizer** (`myai/tokenizer.py`) — Byte-level BPE trained on your own text.
Base vocab is the 256 byte values, then merges are learned by counting adjacent
pairs. Because the base is bytes, *any* UTF-8 text round-trips, including text
the trainer never saw. Special tokens (`<|user|>`, `<|assistant|>`, `<|eot|>`, …)
live above the merges and can never be produced by merging.

```bash
python scripts/prepare_data.py --tinyshakespeare --num-merges 2000
# [tokenizer] trained 2000 merges -> vocab 2262
# [data] 1,059,624 chars -> 353,809 tokens (2.99 chars/token)
```

**2. Pretraining** (`myai/train.py`) — Next-token prediction. Given tokens
`0..t-1`, predict token `t`; cross-entropy over the whole window. This is the
stage that actually teaches the model your data. AdamW, linear warmup then
cosine decay, bf16 autocast, gradient accumulation, gradient clipping,
checkpoint-and-resume.

**3. Instruction tuning** (`myai/sft.py`) — Teaches the *shape* of answering.
Conversations are packed into blocks and loss is computed **only on assistant
tokens**, so the model never learns to imitate the questions.

**4. Serving** (`myai/serve.py`) — Browser chat UI plus `POST /v1/chat/completions`
so existing OpenAI-compatible clients work unchanged.

---

## Architecture

Standard 2024-era small-LLM design, all implemented here:

| Component | Choice | Why |
|---|---|---|
| Norm | RMSNorm (pre-norm) | one less statistic than LayerNorm, same quality |
| Positions | RoPE | no learned position table, extrapolates better |
| FFN | SwiGLU | better than GELU at equal FLOPs |
| Attention | Grouped-query (`n_kv_head ≤ n_head`) | shrinks the KV cache |
| Kernel | `F.scaled_dot_product_attention` | dispatches to FlashAttention when available |
| Decoding | KV cache + sliding window | O(1) per token, never exceeds context |
| Output | Tied embeddings | saves a whole vocab-sized matrix |

`python -m myai.train --list-presets`:

```
smoke              0.4M params  L=2  W=64  H=2  KV=2  ctx=256
rtx3050_4gb       13.8M params  L=6  W=384 H=6  KV=6  ctx=512
rtx3050_8gb       29.9M params  L=8  W=512 H=8  KV=8  ctx=1024
small             60.1M params  L=10 W=640 H=10 KV=10 ctx=1024
medium           100.7M params  L=12 W=768 H=12 KV=4  ctx=2048
```

---

## Sizing for an RTX 3050

The 3050 ships in 4 GB (laptop), 6 GB, and 8 GB (desktop) variants.
`pick_preset_for_vram()` reads the actual VRAM and chooses for you — that is
what `train_rtx3050.sh` uses. If you hit `CUDA out of memory`, in this order:

1. `--grad-checkpoint` — recomputes activations in backward. Costs ~25% speed,
   saves a lot of memory. This is the first thing to try.
2. `--batch-size 4 --grad-accum 16` — same effective batch, a quarter of the
   activation memory.
3. `--block-size 256` — attention memory is quadratic in context length.

bf16 is on by default on Ampere and newer (the 3050 is Ampere). Don't use fp16
unless you have to; bf16 has the range to skip the loss scaler entirely.

## What to realistically expect

Be honest with yourself here, because it saves weeks of confusion.

Your corpus needs roughly **50+ tokens per parameter** before the model learns
structure rather than memorising. For the 13.8M preset that is ~700M tokens. A
3050 pushes maybe 3,000-6,000 tokens/sec at that size, so:

| Effort | Result |
|---|---|
| ~1M tokens, 30 min | Writes plausible-looking text in your domain. Broken grammar. |
| ~50M tokens, ~4 hrs | Coherent sentences, real style, weak reasoning. |
| ~300M tokens, ~20 hrs | Genuinely useful narrow-domain writer. |

**What this will not do:** match ChatGPT. General assistants need trillions of
tokens and thousands of GPU-hours. What you *can* build on a 3050 is a model
that is unmistakably good at one narrow thing — your writing voice, your
codebase's conventions, a single book series, a constructed language. Pick a
narrow corpus and the results are surprisingly satisfying.

Rule of thumb: **a small model on a focused corpus beats a small model on a
broad one**, every time.

---

## Chat data format

`data/chat.jsonl`, one conversation per line:

```json
{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
```

Alpaca format also works: `{"instruction": "...", "input": "...", "output": "..."}`.

A few hundred examples is enough to change behaviour; a few thousand is better.
Write your own — it is the single highest-leverage thing you can do for quality
at this scale.

---

## Tests

```bash
python -m pytest tests/ -q      # 52 passed
```

They cover the parts that silently break models:

- **Causality** — changing a future token must not change earlier logits
- **KV cache == full recompute** — cached decoding must be bit-comparable
- **Gradients reach every parameter** — no silently detached subgraph
- **Loss masking** — supervised targets decode back to exactly the assistant reply
- **Overfit test** — drives the real `train()` and asserts the loss collapses
- **Special-token injection** — user text can't smuggle in control tokens
- **Server endpoints** — live HTTP requests against a real socket

## Verifying your own run

```bash
python -m myai.generate --checkpoint runs/chatbot/best.pt --prompt "your prompt"
python -c "import json;[print(l) for l in open('runs/chatbot/log.jsonl')]" | tail
```

Watch the validation loss. If train loss falls but val loss rises, you are
memorising — get more data or shrink the model.
