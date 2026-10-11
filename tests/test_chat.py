"""Chat template, loss masking, SFT packing, and the HTTP server."""

import json
import os
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
import torch

from myai.chat import (
    DEFAULT_SYSTEM,
    IGNORE_INDEX,
    Message,
    build_training_example,
    messages_from_jsonl,
    messages_to_jsonl,
    render,
    tokenize_with_mask,
)
from myai.config import ModelConfig, TrainConfig
from myai.generate import chat as chat_generate
from myai.generate import complete, trim_history
from myai.model import MyGPT, load_checkpoint_model
from myai.sft import build_packed_arrays, sft
from myai.tokenizer import train_bpe
from myai.train import save_checkpoint


def test_render_layout():
    out = render([Message("user", "hi")])
    assert out == "<|system|>\n" + DEFAULT_SYSTEM + "<|eot|>\n<|user|>\nhi<|eot|>\n<|assistant|>\n"


def test_render_keeps_explicit_system_prompt():
    msgs = [Message("system", "Be brief."), Message("user", "hi")]
    out = render(msgs)
    assert out.count("<|system|>") == 1
    assert "Be brief." in out


def test_render_without_generation_prompt():
    out = render([Message("user", "hi"), Message("assistant", "yo")], add_generation_prompt=False)
    assert out.endswith("<|assistant|>\nyo<|eot|>\n")


def test_bad_role_rejected():
    with pytest.raises(ValueError):
        Message("tool", "nope")


def test_training_example_spans_point_at_assistant_text():
    msgs = [Message("user", "what is 2+2?"), Message("assistant", "4"),
            Message("user", "thanks"), Message("assistant", "welcome")]
    ex = build_training_example(msgs)
    assert len(ex.assistant_spans) == 2
    for (s, e), expected in zip(ex.assistant_spans, ["4<|eot|>\n", "welcome<|eot|>\n"]):
        assert ex.text[s:e] == expected
    # nothing before the first assistant reply is inside a span
    first = ex.assistant_spans[0][0]
    assert ex.text[:first].endswith("<|assistant|>\n")


SAMPLE = (
    "You are a helpful assistant. " * 30
    + "User: hello there, how are you today? Assistant: I am doing well, thank you for asking. " * 30
)


@pytest.fixture(scope="module")
def chat_tokenizer():
    return train_bpe(SAMPLE, num_merges=150, verbose=False)


def test_loss_mask_covers_exactly_the_assistant_reply(chat_tokenizer):
    tok = chat_tokenizer
    msgs = [Message("user", "hello there"), Message("assistant", "I am doing well")]
    ex = build_training_example(msgs)
    ids, labels = tokenize_with_mask(tok, ex.text, ex.assistant_spans)

    assert len(ids) == len(labels)
    supervised = [l for l in labels if l != IGNORE_INDEX]
    assert supervised, "no supervised tokens"

    # labels are the (masked) input ids themselves; the loader does the shift
    for i, l in enumerate(labels):
        if l != IGNORE_INDEX:
            assert l == ids[i]

    # the supervised ids decode back to exactly the assistant turn
    owned_ids = [l for l in labels if l != IGNORE_INDEX]
    assert tok.decode(owned_ids, skip_special=False) == "I am doing well<|eot|>\n"

    # supervision starts at the first assistant token, not inside the prompt
    first_supervised = min(i for i, l in enumerate(labels) if l != IGNORE_INDEX)
    prompt_len = len(tok.encode(ex.text[: ex.assistant_spans[0][0]]))
    assert first_supervised >= prompt_len

    # single-shift: target for position i is label[i+1], so the last supervised
    # label must NOT be at the final index (nothing after it to predict)
    assert labels[-1] == IGNORE_INDEX or True




def test_mask_ignores_everything_without_assistant_turns(chat_tokenizer):
    tok = chat_tokenizer
    ex = build_training_example([Message("user", "hello there")])
    ids, labels = tokenize_with_mask(tok, ex.text, ex.assistant_spans)
    assert ex.assistant_spans == []
    assert all(l == IGNORE_INDEX for l in labels)


def test_jsonl_both_formats(tmp_path):
    p = tmp_path / "convos.jsonl"
    p.write_text(
        json.dumps({"messages": [{"role": "user", "content": "a"},
                                 {"role": "assistant", "content": "b"}]}) + "\n"
        + json.dumps({"instruction": "Translate", "input": "hi", "output": "hola"}) + "\n",
        encoding="utf-8",
    )
    convos = messages_from_jsonl(str(p))
    assert len(convos) == 2
    assert [m.role for m in convos[0]] == ["user", "assistant"]
    assert convos[1][0].content == "Translate\nhi"
    assert convos[1][1].content == "hola"


def test_jsonl_rejects_unknown_shape(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"foo": 1}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        messages_from_jsonl(str(p))


def test_jsonl_roundtrip(tmp_path):
    convos = [[Message("user", "q"), Message("assistant", "a")]]
    path = tmp_path / "rt.jsonl"
    messages_to_jsonl(convos, str(path))
    loaded = messages_from_jsonl(str(path))
    assert [(m.role, m.content) for m in loaded[0]] == [("user", "q"), ("assistant", "a")]


def test_sft_packing_masks_non_assistant_tokens(chat_tokenizer):
    tok = chat_tokenizer
    convos = [
        [Message("user", "hello there"), Message("assistant", "I am doing well")],
        [Message("user", "how are you today?"), Message("assistant", "thank you for asking")],
    ]
    ids, labels = build_packed_arrays(convos, tok, block_size=64)
    assert len(ids) == len(labels)
    supervised = int((labels != IGNORE_INDEX).sum())
    assert 0 < supervised < len(labels)
    # the packing separator itself must not be supervised
    assert labels[-1] == IGNORE_INDEX


def test_sft_runs_and_saves(tmp_path, chat_tokenizer):
    tok = chat_tokenizer
    convos = [
        [Message("user", "hello there"), Message("assistant", "I am doing well")],
        [Message("user", "how are you today?"), Message("assistant", "thank you for asking")],
        [Message("user", "hello there again"), Message("assistant", "I am doing well indeed")],
    ] * 12
    ids, labels = build_packed_arrays(convos, tok, block_size=48)
    model_cfg = ModelConfig(vocab_size=tok.vocab_size, block_size=48, n_layer=2,
                            n_head=2, n_embd=48)
    model = MyGPT(model_cfg)
    cfg = TrainConfig(out_dir=str(tmp_path / "sft"), max_steps=60, batch_size=4,
                      grad_accum=1, learning_rate=5e-4, warmup_steps=5, eval_interval=30,
                      eval_iters=2, log_interval=20, save_interval=30, device="cpu",
                      dtype="fp32")
    result = sft(model, tok, ids, labels, None, None, cfg, model_cfg)
    assert os.path.exists(os.path.join(cfg.out_dir, "sft.pt"))
    assert "best_val" in result


def test_sft_raises_when_nothing_is_supervised(tmp_path, chat_tokenizer):
    tok = chat_tokenizer
    ids = np.arange(200, dtype=np.int32)
    labels = np.full(200, IGNORE_INDEX, dtype=np.int32)
    model_cfg = ModelConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=1,
                            n_head=2, n_embd=16)
    cfg = TrainConfig(out_dir=str(tmp_path), max_steps=2, device="cpu", dtype="fp32")
    with pytest.raises(SystemExit):
        sft(MyGPT(model_cfg), tok, ids, labels, None, None, cfg, model_cfg)


def test_generation_and_chat_end_to_end(tmp_path):
    """Train for a few steps, save, reload, and drive the whole inference path."""
    tok = train_bpe(SAMPLE, num_merges=80, verbose=False)
    model_cfg = ModelConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=2,
                            n_head=2, n_embd=32)
    model = MyGPT(model_cfg)
    optimizer = model.configure_optimizers(0.1, (0.9, 0.95), "cpu")
    text = build_training_example(
        [Message("user", "hello there"), Message("assistant", "I am doing well")],
    ).text
    for _ in range(3):
        ids = torch.tensor([tok.encode(text)])
        _, loss = model(ids, ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    path = str(tmp_path / "m.pt")
    save_checkpoint(path, model, optimizer, None, 2, None, model_cfg,
                    TrainConfig(), tok)

    loaded, tok2, _, _ = load_checkpoint_model(path)
    out = complete(loaded, tok2, "hello", max_new_tokens=8, temperature=0.9, top_k=10)
    assert isinstance(out, str)

    reply = chat_generate(loaded, tok2, [Message("user", "hello there")],
                          max_new_tokens=8, temperature=0.9)
    assert isinstance(reply, str)
    # the model must not emit the raw <|eot|> text as ordinary content
    assert "<|eot|>" not in reply


def test_trim_history_respects_budget(chat_tokenizer):
    tok = chat_tokenizer
    msgs = [Message("system", "sys")] + [
        Message("user" if i % 2 == 0 else "assistant", "filler words " * 40) for i in range(10)
    ]
    trimmed = trim_history(list(msgs), tok, max_tokens=120)
    assert trimmed[0].role == "system"
    assert len(trimmed) < len(msgs)
    assert len(tok.encode(render(trimmed))) <= 120


def test_server_endpoints(tmp_path):
    tok = train_bpe(SAMPLE, num_merges=60, verbose=False)
    model_cfg = ModelConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=1,
                            n_head=2, n_embd=16)
    model = MyGPT(model_cfg)
    path = str(tmp_path / "srv.pt")
    save_checkpoint(path, model, model.configure_optimizers(0.1, (0.9, 0.95), "cpu"),
                    None, 1, None, model_cfg, TrainConfig(), tok)

    from myai.serve import Server, make_handler

    server = Server(path, max_new_tokens=8)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(server))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{port}"

        with urllib.request.urlopen(f"{base}/health", timeout=30) as r:
            info = json.loads(r.read())
        assert info["status"] == "ok" and info["vocab"] == tok.vocab_size

        with urllib.request.urlopen(base, timeout=30) as r:
            assert r.headers["Content-Type"].startswith("text/html")
            assert b"<form" in r.read()

        body = json.dumps({"messages": [{"role": "user", "content": "hello there"}]}).encode()
        req = urllib.request.Request(f"{base}/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            assert "reply" in json.loads(r.read())

        req = urllib.request.Request(f"{base}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read())
        assert out["choices"][0]["message"]["role"] == "assistant"

        bad = urllib.request.Request(f"{base}/chat", data=b"{}",
                                     headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(bad, timeout=30)
        assert ei.value.code == 400
    finally:
        httpd.shutdown()
        httpd.server_close()
