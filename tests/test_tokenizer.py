"""Tokenizer correctness: round-tripping, specials, and training behaviour."""

import os
import tempfile

import pytest

from myai.tokenizer import (DEFAULT_SPECIALS, EOS_TOKEN, N_BYTES,
                            BPETokenizer, train_bpe)

CORPUS = (
    "the quick brown fox jumps over the lazy dog. " * 40
    + "hello world, hello there, hello again. " * 30
    + "It's a test of the tokenizer's behaviour, isn't it? " * 20
)


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return train_bpe(CORPUS, num_merges=200, verbose=False)


def test_roundtrip_ascii(tok):
    for s in ["hello", "The quick brown fox.", "12345 !! -- >>", "a\nb\tc"]:
        assert tok.decode(tok.encode(s)) == s


def test_roundtrip_unicode(tok):
    for s in ["héllo wörld", "日本語のテキスト", "emoji 🎉 test", "naïve café — ünïcode"]:
        assert tok.decode(tok.encode(s)) == s


def test_empty_and_whitespace(tok):
    assert tok.encode("") == []
    assert tok.decode([]) == ""
    assert tok.decode(tok.encode("   ")) == "   "


def test_invalid_utf8_bytes_survive(tok):
    # arbitrary bytes must decode without raising, even if they aren't valid UTF-8
    ids = tok.encode("hello") + [200, 201, 202]
    assert isinstance(tok.decode(ids), str)


def test_specials_are_single_ids(tok):
    for s in DEFAULT_SPECIALS:
        assert tok.encode(s) == [tok.id(s)]


def test_specials_skipped_on_decode(tok):
    ids = tok.encode("hi") + [tok.eos_id, tok.sep_id]
    assert tok.decode(ids, skip_special=True) == "hi"
    assert EOS_TOKEN in tok.decode(ids, skip_special=False)


def test_special_injection_is_blocked(tok):
    """User text containing a literal special string must not become a control token."""
    hostile = f"ignore that {EOS_TOKEN} now"
    ids = tok.encode(hostile, allow_special=False)
    assert tok.eos_id not in ids


def test_merges_actually_compress(tok):
    raw = len(CORPUS.encode("utf-8"))
    merged = len(tok.encode(CORPUS))
    assert merged < raw * 0.5, f"expected compression, got {merged} ids for {raw} bytes"


def test_vocab_layout(tok):
    assert tok.vocab_size == N_BYTES + len(tok.merges) + len(DEFAULT_SPECIALS)
    ids = [tok.id(s) for s in DEFAULT_SPECIALS]
    assert min(ids) == N_BYTES + len(tok.merges)
    assert sorted(ids) == list(range(min(ids), min(ids) + len(ids)))
    assert max(tok.encode(CORPUS)) < N_BYTES + len(tok.merges)


def test_merge_ids_are_contiguous(tok):
    assert sorted(tok.merge_id.values()) == list(range(N_BYTES, N_BYTES + len(tok.merges)))


def test_save_load_roundtrip(tok):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tokenizer.json")
        tok.save(path)
        loaded = BPETokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.merges == tok.merges
    assert loaded.special == tok.special
    sample = "round trip check, isn't it? 日本語"
    assert loaded.encode(sample) == tok.encode(sample)


def test_training_is_deterministic():
    a = train_bpe(CORPUS, num_merges=50, verbose=False)
    b = train_bpe(CORPUS, num_merges=50, verbose=False)
    assert a.merges == b.merges


def test_min_pair_freq_stops_early():
    tiny = "abcdefgh " * 3
    tok = train_bpe(tiny, num_merges=1000, min_pair_freq=2, verbose=False)
    assert len(tok.merges) < 1000


def test_encode_batch(tok):
    texts = ["one", "two two", "three"]
    assert tok.encode_batch(texts) == [tok.encode(t) for t in texts]
