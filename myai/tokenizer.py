"""Byte-level BPE tokenizer, trained from scratch on your own corpus.

No `tokenizers`, no `sentencepiece`, no downloaded vocab files. Training is the
classic Sennrich algorithm: count adjacent symbol pairs, merge the most frequent
one, repeat. The base vocabulary is the 256 byte values, so any UTF-8 text —
including text the trainer never saw — always round-trips.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Pre-tokenisation: BPE merges only ever happen *inside* one of these chunks, so
# a merge can never glue "the" onto the following word. \w is unicode-aware.
PRE_TOKEN_RE = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+""", re.UNICODE)

DEFAULT_SPECIALS: List[str] = [
    "<|endoftext|>",  # document separator / EOS
    "<|pad|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|eot|>",        # end of turn
]

# Named aliases so no other module ever has to retype a special-token string.
EOS_TOKEN = DEFAULT_SPECIALS[0]        # document separator between corpus documents
PAD_TOKEN = DEFAULT_SPECIALS[1]
SYSTEM_TOKEN = DEFAULT_SPECIALS[2]
USER_TOKEN = DEFAULT_SPECIALS[3]
ASSISTANT_TOKEN = DEFAULT_SPECIALS[4]
EOT_TOKEN = DEFAULT_SPECIALS[5]        # end of a chat turn

N_BYTES = 256


class BPETokenizer:
    """Byte-level BPE. `encode(str) -> List[int]`, `decode(List[int]) -> str`."""

    def __init__(
        self,
        merges: Sequence[Tuple[int, int]],
        specials=None,
    ):
        self.merges: List[Tuple[int, int]] = [tuple(m) for m in merges]  # type: ignore[misc]
        self.merge_rank: Dict[Tuple[int, int], int] = {m: i for i, m in enumerate(self.merges)}
        self.merge_id: Dict[Tuple[int, int], int] = {
            m: N_BYTES + i for i, m in enumerate(self.merges)
        }
        base = N_BYTES + len(self.merges)
        if specials is None:
            names = list(DEFAULT_SPECIALS)
            self.special: Dict[str, int] = {s: base + i for i, s in enumerate(names)}
        elif isinstance(specials, dict):
            self.special = {k: int(v) for k, v in specials.items()}
        else:  # a plain list of names: ids are assigned right after the merges
            self.special = {s: base + i for i, s in enumerate(specials)}
        self.special_ids = set(self.special.values())
        self.id_to_special: Dict[int, str] = {v: k for k, v in self.special.items()}
        self._special_re = re.compile(
            "(" + "|".join(re.escape(s) for s in self.special) + ")"
        ) if self.special else None
        self._chunk_cache: Dict[str, List[int]] = {}
        # every id -> the exact bytes it stands for, built by replaying the merges
        self.id_to_bytes: Dict[int, bytes] = {i: bytes([i]) for i in range(N_BYTES)}
        for rank, (a, b) in enumerate(self.merges):
            self.id_to_bytes[N_BYTES + rank] = self.id_to_bytes[a] + self.id_to_bytes[b]

    # -- convenience ids ---------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return N_BYTES + len(self.merges) + len(self.special)

    def id(self, special: str) -> int:
        return self.special[special]

    @property
    def eos_id(self) -> int:
        """End of a chat turn; falls back to the document separator."""
        return self.special.get(EOT_TOKEN, self.sep_id)

    @property
    def sep_id(self) -> int:
        """Document separator — what prepare_data.py puts between documents."""
        return self.special[EOS_TOKEN]

    @property
    def pad_id(self) -> int:
        return self.special.get("<|pad|>", 0)

    # -- encode ------------------------------------------------------------
    def _encode_chunk(self, chunk: str) -> List[int]:
        cached = self._chunk_cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            best_rank, best_i = None, -1
            for i in range(len(ids) - 1):
                rank = self.merge_rank.get((ids[i], ids[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_i = rank, i
            if best_i < 0:
                break
            new_id = self.merge_id[(ids[best_i], ids[best_i + 1])]
            ids = ids[:best_i] + [new_id] + ids[best_i + 2 :]
        if len(self._chunk_cache) < 200_000:
            self._chunk_cache[chunk] = ids
        return ids

    def encode(self, text: str, allow_special: bool = True) -> List[int]:
        """Text -> token ids. Literal special-token strings become single ids."""
        if not text:
            return []
        out: List[int] = []
        if allow_special and self._special_re is not None:
            for part in self._special_re.split(text):
                if not part:
                    continue
                if part in self.special:
                    out.append(self.special[part])
                else:
                    for chunk in PRE_TOKEN_RE.findall(part):
                        out.extend(self._encode_chunk(chunk))
        else:
            # treat special strings as ordinary text: strip them so they can
            # never be smuggled into the token stream
            if self._special_re is not None:
                text = self._special_re.sub(" ", text)
            for chunk in PRE_TOKEN_RE.findall(text):
                out.extend(self._encode_chunk(chunk))
        return out

    def encode_batch(self, texts: Iterable[str], allow_special: bool = True) -> List[List[int]]:
        return [self.encode(t, allow_special) for t in texts]

    # -- decode ------------------------------------------------------------
    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        """Token ids -> text.

        Byte ids and merged ids both resolve through `id_to_bytes`; special tokens
        are dropped by default and expanded to their literal string otherwise.
        Unknown ids are ignored rather than raising, so a corrupt generation can
        never crash inference.
        """
        buf = bytearray()
        pieces: List[str] = []

        def flush() -> None:
            if buf:
                pieces.append(bytes(buf).decode("utf-8", errors="replace"))
                buf.clear()

        for i in ids:
            if i in self.special_ids:
                if not skip_special:
                    flush()
                    pieces.append(self.id_to_special[i])
                continue
            b = self.id_to_bytes.get(i)
            if b is not None:
                buf.extend(b)
        flush()
        return "".join(pieces)

    def decode_token(self, token_id: int) -> str:
        """Decode a single id, special tokens included."""
        if token_id in self.special_ids:
            return self.id_to_special[token_id]
        b = self.id_to_bytes.get(token_id)
        return b.decode("utf-8", errors="replace") if b is not None else ""

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "type": "byte-level-bpe",
            "merges": [list(m) for m in self.merges],
            "special": self.special,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BPETokenizer":
        return cls(
            merges=[(int(a), int(b)) for a, b in d["merges"]],
            specials={k: int(v) for k, v in d["special"].items()},
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def __repr__(self) -> str:
        return f"BPETokenizer(merges={len(self.merges)}, vocab={self.vocab_size})"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _apply_merge(seq: List[int], a: int, b: int, new_id: int) -> List[int]:
    out: List[int] = []
    i = 0
    n = len(seq)
    while i < n:
        if i < n - 1 and seq[i] == a and seq[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


class _BPETrainer:
    """Incremental merge bookkeeping: each merge only touches words containing it."""

    def __init__(self, counter: "Counter[str]"):
        self.freq = counter
        self.seqs: Dict[str, List[int]] = {w: list(w.encode("utf-8")) for w in counter}
        self.pair_count: Counter = Counter()
        self.pair_words: Dict[Tuple[int, int], set] = defaultdict(set)
        for w in self.seqs:
            self._index(w)

    def _pairs(self, seq: List[int]) -> Iterable[Tuple[int, int]]:
        for i in range(len(seq) - 1):
            yield (seq[i], seq[i + 1])

    def _index(self, w: str) -> None:
        f = self.freq[w]
        for p in self._pairs(self.seqs[w]):
            self.pair_count[p] += f
            self.pair_words[p].add(w)

    def _unindex(self, w: str) -> None:
        f = self.freq[w]
        for p in self._pairs(self.seqs[w]):
            self.pair_count[p] -= f
            if self.pair_count[p] <= 0:
                del self.pair_count[p]
            self.pair_words[p].discard(w)

    def best_pair(self) -> Optional[Tuple[int, int]]:
        if not self.pair_count:
            return None
        # GPT-2 tie-break: highest count, then the lexicographically larger pair
        return max(self.pair_count.items(), key=lambda kv: (kv[1], kv[0]))[0]

    def count(self, pair: Tuple[int, int]) -> int:
        return self.pair_count.get(pair, 0)

    def do_merge(self, pair: Tuple[int, int], new_id: int) -> None:
        a, b = pair
        for w in list(self.pair_words[pair]):
            self._unindex(w)
            self.seqs[w] = _apply_merge(self.seqs[w], a, b, new_id)
            self._index(w)
        self.pair_count.pop(pair, None)
        self.pair_words.pop(pair, None)


def train_bpe(
    text: str,
    num_merges: int = 2048,
    min_pair_freq: int = 2,
    specials: Optional[List[str]] = None,
    verbose: bool = True,
    log_every: int = 256,
) -> BPETokenizer:
    """Train byte-level BPE on `text`. `num_merges` controls final vocab size."""
    specials = DEFAULT_SPECIALS if specials is None else specials

    # Special strings must never be merged into, so split them out of the corpus.
    if specials:
        special_re = re.compile("(" + "|".join(re.escape(s) for s in specials) + ")")
        text = special_re.sub(" ", text)

    counter: Counter = Counter(PRE_TOKEN_RE.findall(text))
    if verbose:
        n_chunks = sum(counter.values())
        print(f"[tokenizer] {len(text):,} chars -> {n_chunks:,} pre-tokens, "
              f"{len(counter):,} unique")

    trainer = _BPETrainer(counter)
    merges: List[Tuple[int, int]] = []
    # id -> the bytes it stands for, so we can print a human-readable merge log
    # even once ids exceed 255 (bytes() alone would raise there)
    piece: Dict[int, bytes] = {i: bytes([i]) for i in range(N_BYTES)}
    for step in range(num_merges):
        pair = trainer.best_pair()
        if pair is None:
            if verbose:
                print(f"[tokenizer] no pairs left after {step} merges")
            break
        pair_count = trainer.count(pair)
        if pair_count < min_pair_freq:
            if verbose:
                print(f"[tokenizer] stopping: best pair count {pair_count} "
                      f"< min_pair_freq {min_pair_freq} at merge {step}")
            break
        merges.append(pair)
        new_id = N_BYTES + step
        piece[new_id] = piece[pair[0]] + piece[pair[1]]
        trainer.do_merge(pair, new_id)
        if verbose and (step + 1) % log_every == 0:
            print(f"[tokenizer] merge {step + 1}/{num_merges} "
                  f"({piece[new_id].decode('utf-8', 'replace')!r}, count={pair_count})")

    tok = BPETokenizer(merges=merges, specials=specials)
    if verbose:
        print(f"[tokenizer] trained {len(merges)} merges -> vocab {tok.vocab_size}")
    return tok


def train_bpe_from_files(
    paths: Sequence[str],
    num_merges: int = 2048,
    max_chars: int = 40_000_000,
    **kwargs,
) -> BPETokenizer:
    """Read text files, concatenate up to `max_chars`, and train on the result."""
    chunks: List[str] = []
    total = 0
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            data = f.read(max_chars - total)
        chunks.append(data)
        total += len(data)
        if total >= max_chars:
            break
    return train_bpe("".join(chunks), num_merges=num_merges, **kwargs)
