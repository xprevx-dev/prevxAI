"""HTTP server: a browser chat UI plus an OpenAI-compatible API.

Standard library only — no FastAPI, no uvicorn.

    python -m myai.serve --checkpoint runs/chatbot/sft.pt --port 8000

Endpoints:
    GET  /                       browser chat UI
    POST /chat                   {"messages": [...]} -> {"reply": "..."}
    POST /v1/chat/completions    OpenAI-compatible, so existing clients work
    GET  /health                 model info
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional

import torch

from myai.chat import DEFAULT_SYSTEM, Message, ROLES
from myai.generate import chat as chat_generate
from myai.model import load_checkpoint_model

_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>myai</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; height: 100vh; display: flex; flex-direction: column;
    background: #0d1117; color: #e6edf3;
    font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  }
  header {
    padding: 12px 18px; border-bottom: 1px solid #21262d;
    display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
  }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  header .meta { color: #7d8590; font-size: 12px; }
  #log { flex: 1; overflow-y: auto; padding: 18px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 78%; padding: 10px 14px; border-radius: 12px; white-space: pre-wrap; word-wrap: break-word; }
  .user { align-self: flex-end; background: #1f6feb; }
  .assistant { align-self: flex-start; background: #161b22; border: 1px solid #21262d; }
  .system-note { align-self: center; color: #7d8590; font-size: 12px; }
  form { display: flex; gap: 8px; padding: 12px 18px; border-top: 1px solid #21262d; }
  textarea {
    flex: 1; resize: none; height: 46px; padding: 12px; border-radius: 10px;
    border: 1px solid #30363d; background: #0d1117; color: inherit; font: inherit;
  }
  textarea:focus { outline: 1px solid #1f6feb; }
  button {
    padding: 0 20px; border-radius: 10px; border: 1px solid #1f6feb;
    background: #1f6feb; color: #fff; font: inherit; font-weight: 600; cursor: pointer;
  }
  button:disabled { opacity: .5; cursor: default; }
  button.ghost { background: transparent; border-color: #30363d; color: #7d8590; }
</style>
</head>
<body>
<header>
  <h1>myai</h1>
  <span class="meta" id="meta">loading&hellip;</span>
  <button class="ghost" id="clear" type="button">Clear</button>
</header>
<div id="log"></div>
<form id="form">
  <textarea id="input" placeholder="Say something&hellip; (Enter to send, Shift+Enter for a newline)"></textarea>
  <button id="send" type="submit">Send</button>
</form>
<script>
const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('input');
const send = document.getElementById('send');
let history = [];

function add(role, text) {
  const el = document.createElement('div');
  el.className = 'msg ' + role;
  el.textContent = text;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

fetch('/health').then(r => r.json()).then(info => {
  document.getElementById('meta').textContent =
    info.params_m + 'M params · vocab ' + info.vocab + ' · context ' + info.block_size +
    ' · step ' + info.step;
});

document.getElementById('clear').onclick = () => { history = []; log.innerHTML = ''; };

form.onsubmit = async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  add('user', text);
  history.push({role: 'user', content: text});
  send.disabled = true;
  const pending = add('assistant', '…');
  try {
    const r = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({messages: history}),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || r.statusText);
    pending.textContent = data.reply;
    history.push({role: 'assistant', content: data.reply});
  } catch (err) {
    pending.textContent = 'Error: ' + err.message;
  }
  send.disabled = false;
  input.focus();
};

input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.onsubmit(e); }
});
</script>
</body>
</html>
"""


class Server:
    """Holds the model and serialises inference behind a lock."""

    def __init__(self, checkpoint: str, device: Optional[str] = None, dtype=None,
                 system: str = DEFAULT_SYSTEM, max_new_tokens: int = 256):
        self.model, self.tokenizer, _, self.ckpt = load_checkpoint_model(
            checkpoint, device=device or "cpu", dtype=dtype
        )
        self.device = str(next(self.model.parameters()).device)
        self.lock = threading.Lock()
        self.system = system
        self.max_new_tokens = max_new_tokens
        print(f"[serve] model on {self.device}: {self.model.num_params() / 1e6:.1f}M params, "
              f"step {self.ckpt.get('step')}, vocab {self.tokenizer.vocab_size}")

    def info(self) -> dict:
        return {
            "status": "ok",
            "params_m": round(self.model.num_params() / 1e6, 2),
            "vocab": self.tokenizer.vocab_size,
            "block_size": self.model.cfg.block_size,
            "step": self.ckpt.get("step"),
            "best_val": self.ckpt.get("best_val"),
            "device": self.device,
        }

    def reply(self, messages: List[Message], temperature: float = 0.7,
              max_new_tokens: Optional[int] = None) -> str:
        with self.lock:
            return chat_generate(
                self.model, self.tokenizer, messages,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                temperature=temperature, system=self.system, device=self.device,
            )


def make_handler(server: Server):
    class Handler(BaseHTTPRequestHandler):
        server_version = "myai/0.1"

        def log_message(self, fmt, *args):  # quieter access log
            print("[serve] " + fmt % args)

        def _send(self, code: int, body: bytes, ctype: str = "application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode("utf-8"))

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, _HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/health":
                self._json(200, server.info())
            else:
                self._json(404, {"error": "not found"})

        def _read_json(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n == 0:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8"))

        def do_POST(self):
            try:
                body = self._read_json()
            except json.JSONDecodeError as e:
                self._json(400, {"error": f"invalid JSON: {e}"})
                return

            raw = body.get("messages")
            if not isinstance(raw, list) or not raw:
                self._json(400, {"error": "'messages' must be a non-empty list"})
                return
            try:
                msgs = [Message(m["role"], m["content"]) for m in raw]
            except (KeyError, ValueError, TypeError) as e:
                self._json(400, {"error": f"bad message: {e}; roles are {list(ROLES)}"})
                return

            t0 = time.time()
            try:
                reply = server.reply(msgs, temperature=float(body.get("temperature", 0.7)),
                                     max_new_tokens=body.get("max_tokens"))
            except Exception as e:  # keep the server alive on a bad request
                self._json(500, {"error": f"{type(e).__name__}: {e}"})
                return
            dt = time.time() - t0

            if self.path.startswith("/v1/chat/completions"):
                self._json(200, {
                    "id": f"myai-{int(t0 * 1000)}",
                    "object": "chat.completion",
                    "created": int(t0),
                    "model": "myai",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }],
                    "usage": {"completion_tokens": len(server.tokenizer.encode(reply)),
                              "latency_s": round(dt, 3)},
                })
            else:
                self._json(200, {"reply": reply, "latency_s": round(dt, 3)})

    return Handler


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Serve the model over HTTP.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None, choices=[None, "bf16", "fp16", "fp32"])
    p.add_argument("--system", default=DEFAULT_SYSTEM)
    p.add_argument("--max-new-tokens", type=int, default=256)
    args = p.parse_args(argv)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(args.dtype)
    server = Server(args.checkpoint, device=args.device, dtype=dtype,
                    system=args.system, max_new_tokens=args.max_new_tokens)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(server))
    print(f"[serve] listening on http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] shutting down")
        httpd.shutdown()


if __name__ == "__main__":
    main()
