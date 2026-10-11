"""Inference: raw text completion, chat, and an interactive REPL.

    python -m myai.generate --checkpoint runs/chatbot/best.pt --prompt "Once upon"
    python -m myai.generate --checkpoint runs/chatbot/sft.pt --chat
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

import torch

from myai.chat import DEFAULT_SYSTEM, Message, render
from myai.model import MyGPT, load_checkpoint_model
from myai.tokenizer import BPETokenizer


@torch.no_grad()
def complete(
    model: MyGPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    stop_at_eos: bool = False,
    device: str = "cpu",
) -> str:
    """Continue `prompt` and return only the newly generated text."""
    model.eval()
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [tokenizer.eos_id]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_id=tokenizer.eos_id if stop_at_eos else None,
    )
    new_ids = out[0].tolist()[len(ids):]
    if stop_at_eos and tokenizer.eos_id in new_ids:
        new_ids = new_ids[: new_ids.index(tokenizer.eos_id)]
    return tokenizer.decode(new_ids)


@torch.no_grad()
def chat(
    model: MyGPT,
    tokenizer: BPETokenizer,
    messages: Sequence[Message],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    system: Optional[str] = DEFAULT_SYSTEM,
    device: str = "cpu",
) -> str:
    """Apply the chat template, generate the assistant turn, return its text."""
    prompt = render(list(messages), add_generation_prompt=True, system=system)
    reply = complete(
        model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature,
        top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
        stop_at_eos=True, device=device,
    )
    return reply.strip()


def trim_history(messages: List[Message], tokenizer: BPETokenizer, max_tokens: int) -> List[Message]:
    """Drop the oldest turns (keeping the system prompt) until we fit the context."""
    def fits(msgs: List[Message]) -> bool:
        return len(tokenizer.encode(render(msgs, add_generation_prompt=True))) <= max_tokens

    while messages and not fits(messages):
        # never drop a leading system message; drop the oldest non-system turn
        for i, m in enumerate(messages):
            if m.role != "system":
                messages.pop(i)
                break
        else:
            break
    return messages


def repl(
    model: MyGPT,
    tokenizer: BPETokenizer,
    device: str,
    system: str = DEFAULT_SYSTEM,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> None:
    """Interactive chat. Commands: /reset, /system <text>, /raw <prompt>, /quit."""
    history: List[Message] = []
    budget = max(64, model.cfg.block_size - max_new_tokens - 8)
    print(f"myai chat — context {model.cfg.block_size} tokens, /reset to clear, /quit to exit\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit", "/q"):
            break
        if user == "/reset":
            history = []
            print("[cleared]\n")
            continue
        if user.startswith("/system "):
            system = user[len("/system "):].strip()
            history = [m for m in history if m.role != "system"]
            print(f"[system prompt set]\n")
            continue
        if user.startswith("/raw "):
            print("bot> " + complete(model, tokenizer, user[5:], device=device) + "\n")
            continue

        history.append(Message("user", user))
        history = trim_history(history, tokenizer, budget)
        reply = chat(model, tokenizer, history, max_new_tokens=max_new_tokens,
                     temperature=temperature, system=system, device=device)
        history.append(Message("assistant", reply))
        print(f"bot> {reply}\n")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Generate text or chat with a trained model.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default=None)
    p.add_argument("--chat", action="store_true", help="Interactive chat REPL")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--system", default=DEFAULT_SYSTEM)
    p.add_argument("--device", default=None, help="Default: where the checkpoint lives")
    p.add_argument("--dtype", default=None, choices=[None, "bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(args.dtype)
    model, tokenizer, _, ckpt = load_checkpoint_model(
        args.checkpoint, device=args.device or "cpu", dtype=dtype
    )
    device = next(model.parameters()).device
    step = ckpt.get("step")
    print(f"[generate] loaded {args.checkpoint} "
          f"({model.num_params() / 1e6:.1f}M params, step {step}, vocab {tokenizer.vocab_size})",
          file=sys.stderr)

    if args.chat:
        repl(model, tokenizer, str(device), system=args.system,
             max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        return

    if not args.prompt:
        print("[generate] no --prompt and no --chat; reading stdin", file=sys.stderr)
        args.prompt = sys.stdin.read()

    out = complete(
        model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty, device=str(device),
    )
    print(args.prompt + out)


if __name__ == "__main__":
    main()
