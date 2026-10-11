#!/usr/bin/env bash
# One-command training run, sized to whatever GPU you actually have.
#
#   ./scripts/train_rtx3050.sh                      # auto-size, TinyShakespeare
#   ./scripts/train_rtx3050.sh ~/my_corpus          # auto-size, your own text
#   PRESET=small ./scripts/train_rtx3050.sh         # force a preset
#   STEPS=10000 ./scripts/train_rtx3050.sh
#
# Safe to interrupt: checkpoints land in runs/chatbot/last.pt and
# `--resume` picks up exactly where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."

CORPUS="${1:-}"
DATA="${DATA:-data}"
OUT="${OUT:-runs/chatbot}"
STEPS="${STEPS:-3000}"
CHAT_DATA="${CHAT_DATA:-$DATA/chat.jsonl}"

python3 - "$CORPUS" <<'PY'
import sys, shutil, subprocess
import torch

if not shutil.which("nvidia-smi"):
    print("[setup] nvidia-smi not found — is the NVIDIA driver installed?")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    vram = p.total_memory / 1024**3
    print(f"[setup] GPU: {p.name}, {vram:.1f} GB VRAM, CUDA {torch.version.cuda}")
    print(f"[setup] bf16 supported: {torch.cuda.is_bf16_supported()}")
else:
    print("[setup] WARNING: PyTorch sees no CUDA device. You installed the CPU wheel?")
    print("[setup]   pip install torch --index-url https://download.pytorch.org/whl/cu121")
PY

if [[ -z "${PRESET:-}" ]]; then
  PRESET=$(python3 -c "from myai.config import pick_preset_for_vram; print(pick_preset_for_vram())")
fi
echo "[setup] using preset: $PRESET"
python3 -m myai.train --list-presets | grep -E "^$PRESET" || true

# ---- data ------------------------------------------------------------------
if [[ ! -f "$DATA/train.bin" ]]; then
  if [[ -n "$CORPUS" ]]; then
    echo "[setup] tokenising your corpus: $CORPUS"
    python3 scripts/prepare_data.py --input "$CORPUS" --out-dir "$DATA"
  else
    echo "[setup] no corpus given; using TinyStories (simple GPT-generated stories)"
    python3 scripts/prepare_data.py --tinystories ${TS_MB:+--ts-mb $TS_MB} --out-dir "$DATA"
  fi
fi

# ---- pretrain --------------------------------------------------------------
RESUME=""
[[ -f "$OUT/last.pt" ]] && RESUME="--resume $OUT/last.pt" && echo "[setup] resuming from $OUT/last.pt"

# 4 GB cards (RTX 3050 laptop) need the memory-saving trio: bf16, gradient
# checkpointing, and accumulation instead of a big batch. All are safe defaults.
python3 -m myai.train \
  --preset "$PRESET" \
  --data-dir "$DATA" \
  --out-dir "$OUT" \
  --max-steps "$STEPS" \
  --batch-size 8 \
  --grad-accum 8 \
  --grad-checkpoint \
  --device cuda \
  --dtype bf16 \
  $RESUME

# ---- fine-tune into a chatbot ---------------------------------------------
if [[ ! -f "$CHAT_DATA" ]]; then
  echo "[setup] generating a starter chat set -> $CHAT_DATA"
  python3 scripts/make_chat_data.py --out "$CHAT_DATA" --n 800
fi

if [[ -f "$CHAT_DATA" ]]; then
  echo "[setup] instruction fine-tuning on $CHAT_DATA"
  python3 -m myai.sft --checkpoint "$OUT/best.pt" --data "$CHAT_DATA" --out-dir "$OUT"
  CKPT="$OUT/sft.pt"
else
  echo "[setup] no $CHAT_DATA — skipping fine-tuning"
  echo "[setup] create one (JSONL, one conversation per line):"
  echo '        {"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}'
  CKPT="$OUT/best.pt"
fi

echo
echo "[done] chat with it:  python -m myai.generate --checkpoint $CKPT --chat"
echo "[done] or serve it:   python -m myai.serve --checkpoint $CKPT --port 8000"
