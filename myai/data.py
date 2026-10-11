"""Data pipeline: raw text -> token ids on disk -> random batches.

Tokenised data lives in a memory-mapped .bin file so a corpus far larger than
RAM still trains fine — the OS pages in only what each batch needs.
"""

from __future__ import annotations

import glob
import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from myai.tokenizer import BPETokenizer


def find_text_files(root: str, patterns: Sequence[str] = ("**/*.txt", "**/*.md")) -> List[str]:
    """Recursively collect text files under `root` (a plain file path works too)."""
    if os.path.isfile(root):
        return [root]
    out: List[str] = []
    for pat in patterns:
        out.extend(glob.glob(os.path.join(root, pat), recursive=True))
    return sorted(set(out))


def read_texts(paths: Sequence[str], max_chars: Optional[int] = None) -> str:
    parts: List[str] = []
    total = 0
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            data = f.read()
        parts.append(data)
        total += len(data)
        if max_chars and total >= max_chars:
            break
    return "\n\n".join(parts)


def tokenize_to_bin(
    text: str,
    tokenizer: BPETokenizer,
    out_path: str,
    append_eos: bool = True,
    verbose: bool = True,
) -> int:
    """Tokenise `text` and write the id array to `out_path`. Returns token count."""
    ids = tokenizer.encode(text)
    if append_eos:
        ids = ids + [tokenizer.sep_id]
    # uint16 covers up to 65535 merges — every preset fits, but stay correct
    dtype = np.uint16 if tokenizer.vocab_size < 65535 else np.uint32
    arr = np.array(ids, dtype=dtype)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    arr.tofile(out_path)
    if verbose:
        ratio = len(text) / max(len(ids), 1)
        print(f"[data] {len(text):,} chars -> {len(ids):,} tokens "
              f"({ratio:.2f} chars/token, {dtype.__name__}) -> {out_path}")
    return len(ids)


class BinDataset(Dataset):
    """Random windows over a memory-mapped token stream."""

    def __init__(self, bin_path: str, block_size: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        if len(self.data) <= block_size:
            raise ValueError(
                f"{bin_path} holds {len(self.data)} tokens but block_size={block_size}. "
                "Use a smaller block_size or a larger corpus."
            )

    def __len__(self) -> int:
        return max(1, (len(self.data) - 1) // self.block_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = (idx * self.block_size) % (len(self.data) - self.block_size - 1)
        chunk = self.data[start : start + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def random_batch(data: np.memmap, block_size: int, batch_size: int, device: str):
    """Fast path for the training loop: sample windows straight from the memmap."""
    hi = len(data) - block_size - 1
    ix = np.random.randint(0, hi, size=(batch_size,))
    x = np.stack([data[i : i + block_size].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block_size].astype(np.int64) for i in ix])
    return torch.from_numpy(x).to(device, non_blocking=True), torch.from_numpy(y).to(
        device, non_blocking=True
    )


def load_bin(path: str) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


def write_meta(data_dir: str, **kwargs) -> None:
    meta_path = os.path.join(data_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    meta.update(kwargs)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def read_meta(data_dir: str) -> dict:
    meta_path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)
