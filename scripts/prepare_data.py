#!/usr/bin/env python3
"""Turn raw text into the artefacts training needs.

Produces, inside --out-dir:
    tokenizer.json   the BPE vocab, trained on your own corpus
    train.bin        uint16 token ids
    val.bin          uint16 token ids (held out)
    meta.json        provenance: sources, counts, merge count

Examples:
    # tiny public corpus, good for a first end-to-end run
    python scripts/prepare_data.py --tinyshakespeare

    # your own writing / notes / codebase
    python scripts/prepare_data.py --input ~/notes --num-merges 4000
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myai.data import find_text_files, read_texts, tokenize_to_bin, write_meta  # noqa: E402
from myai.tokenizer import train_bpe  # noqa: E402

TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
TINYSTORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
)


def download(url: str, dest: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    print(f"[data] downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
        f.write(r.read())
    size = os.path.getsize(dest)
    print(f"[data] saved {size / 1024:.0f} KB -> {dest}")
    return dest


def download_range(url: str, dest: str, max_bytes: int = 0) -> str:
    """Download a file, optionally only its first `max_bytes` (HTTP Range).

    TinyStories is 1.9 GB; a laptop demo does not need all of it, and a Range
    request grabs a prefix without touching the rest.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    headers = {"Range": f"bytes=0-{max_bytes - 1}"} if max_bytes > 0 else {}
    what = f"first {max_bytes / 1e6:.0f} MB of " if max_bytes else ""
    print(f"[data] downloading {what}{url}")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    size = os.path.getsize(dest)
    print(f"[data] saved {size / 1e6:.1f} MB -> {dest}")
    return dest


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Tokenise a corpus for training.")
    p.add_argument("--input", default=None, help="Text file or directory of .txt/.md files")
    p.add_argument("--tinyshakespeare", action="store_true",
                   help="Download the 1 MB TinyShakespeare corpus instead")
    p.add_argument("--tinystories", action="store_true",
                   help="Download TinyStories (GPT-generated simple stories)")
    p.add_argument("--ts-mb", type=int, default=0,
                   help="With --tinystories: only fetch the first N MB (0 = full 1.9 GB)")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--num-merges", type=int, default=2000,
                   help="Final vocab = 256 + merges + 6 specials")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--max-chars", type=int, default=40_000_000)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args(argv)

    if args.tinystories:
        dest = os.path.join(args.out_dir, "tinystories.txt")
        if os.path.exists(dest) and args.ts_mb and os.path.getsize(dest) >= args.ts_mb * 1_000_000:
            print(f"[data] reusing cached {dest}")
        else:
            download_range(TINYSTORIES_URL, dest, max_bytes=args.ts_mb * 1_000_000)
        paths = [dest]
    elif args.tinyshakespeare:
        path = download(TINYSHAKESPEARE_URL, os.path.join(args.out_dir, "tinyshakespeare.txt"))
        paths = [path]
    elif args.input:
        paths = find_text_files(args.input)
        if not paths:
            raise SystemExit(f"[data] no text files found under {args.input}")
    else:
        raise SystemExit("[data] pass --input <path> or --tinyshakespeare")

    print(f"[data] {len(paths)} source file(s)")
    text = read_texts(paths, max_chars=args.max_chars)
    print(f"[data] {len(text):,} chars loaded")

    # hold out a slice of the raw text so train and val never share documents
    random.seed(args.seed)
    cut = max(1, int(len(text) * (1 - args.val_frac)))
    train_text, val_text = text[:cut], text[cut:]
    if len(val_text) < 100:
        print("[data] corpus too small to hold out a split; using a 2k-char tail instead")
        train_text, val_text = text[:-2000], text[-2000:]

    tokenizer = train_bpe(train_text, num_merges=args.num_merges)
    tokenizer.save(os.path.join(args.out_dir, "tokenizer.json"))
    print(f"[data] tokenizer vocab {tokenizer.vocab_size} -> "
          f"{os.path.join(args.out_dir, 'tokenizer.json')}")

    n_train = tokenize_to_bin(train_text, tokenizer, os.path.join(args.out_dir, "train.bin"))
    n_val = tokenize_to_bin(val_text, tokenizer, os.path.join(args.out_dir, "val.bin"))

    write_meta(
        args.out_dir,
        sources=[os.path.abspath(p) for p in paths][:50],
        n_sources=len(paths),
        chars=len(text),
        num_merges=len(tokenizer.merges),
        vocab_size=tokenizer.vocab_size,
        train_tokens=n_train,
        val_tokens=n_val,
    )
    print(f"[data] done. train={n_train:,} tokens val={n_val:,} tokens in {args.out_dir}/")

    # a usable rule of thumb: you want >= ~50 tokens per parameter for the model
    # to actually learn something rather than memorise
    print(f"[data] next: python -m myai.train --preset rtx3050_4gb "
          f"--data-dir {args.out_dir} --out-dir runs/chatbot")


if __name__ == "__main__":
    main()
