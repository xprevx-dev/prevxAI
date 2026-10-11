"""Chat formatting.

One template is used in exactly three places, and it must never drift between
them: building SFT training examples (sft.py), the REPL (chat.py), and the HTTP
server (serve.py).

    <|system|>
    You are helpful.<|eot|>
    <|user|>
    hi<|eot|>
    <|assistant|>
    hello!<|eot|>

The model is trained to emit `<|eot|>` to end its turn; generation stops there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from myai.tokenizer import BPETokenizer

DEFAULT_SYSTEM = "You are a helpful assistant."
IGNORE_INDEX = -100
ROLES = ("system", "user", "assistant")


@dataclass
class Message:
    role: str          # system | user | assistant
    content: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}, expected one of {ROLES}")


def render(
    messages: Sequence[Message],
    add_generation_prompt: bool = True,
    system: Optional[str] = DEFAULT_SYSTEM,
) -> str:
    """Messages -> the exact string the model sees at inference time."""
    parts: List[str] = []
    if system and not any(m.role == "system" for m in messages):
        parts.append(f"<|system|>\n{system}<|eot|>\n")
    for m in messages:
        parts.append(f"<|{m.role}|>\n{m.content}<|eot|>\n")
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "".join(parts)


@dataclass
class ChatExample:
    """One training example, with the assistant-reply character spans recorded."""

    text: str
    assistant_spans: List[Tuple[int, int]] = field(default_factory=list)


def build_training_example(
    messages: Sequence[Message],
    system: Optional[str] = DEFAULT_SYSTEM,
) -> ChatExample:
    """Concatenate a conversation and record where each assistant reply lands.

    Loss masking matters: we want the model to learn to *answer*, not to learn
    to produce the user's questions.
    """
    parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    if system and not any(m.role == "system" for m in messages):
        parts.append(f"<|system|>\n{system}<|eot|>\n")
    for m in messages:
        parts.append(f"<|{m.role}|>\n")
        if m.role == "assistant":
            start = sum(len(p) for p in parts)
            parts.append(m.content + "<|eot|>\n")
            spans.append((start, sum(len(p) for p in parts)))
        else:
            parts.append(m.content + "<|eot|>\n")
    return ChatExample(text="".join(parts), assistant_spans=spans)


def _char_ownership(text: str, spans: Sequence[Tuple[int, int]]) -> List[bool]:
    owner = [False] * len(text)
    for s, e in spans:
        for i in range(s, min(e, len(text))):
            owner[i] = True
    return owner


def tokenize_with_mask(
    tokenizer: BPETokenizer,
    text: str,
    spans: Sequence[Tuple[int, int]],
    ignore_index: int = IGNORE_INDEX,
) -> Tuple[List[int], List[int]]:
    """Tokenise `text`, returning (input_ids, labels).

    Labels are the input ids shifted by one (the standard next-token target) with
    every position outside an assistant span set to `ignore_index`, so
    cross_entropy skips them.

    The text is encoded in maximal runs of uniform ownership: ids and labels are
    produced from the same slicing, so they can never disagree.
    """
    owner = _char_ownership(text, spans)
    ids: List[int] = []
    owned: List[bool] = []
    cursor = 0
    while cursor < len(text):
        nxt = cursor + 1
        while nxt < len(text) and owner[nxt] == owner[cursor]:
            nxt += 1
        piece_ids = tokenizer.encode(text[cursor:nxt], allow_special=True)
        ids.extend(piece_ids)
        owned.extend([owner[cursor]] * len(piece_ids))
        cursor = nxt

    # labels[i] is the token id itself, kept only when that token is one we want
    # the model to learn to produce. The single next-token shift (target for
    # position i = label[i+1]) is applied by the data loader, exactly as in
    # pretraining — shifting here too would teach the model to predict two
    # tokens ahead (the classic off-by-one that yields word-soup).
    labels: List[int] = []
    for i in range(len(ids)):
        labels.append(ids[i] if owned[i] else ignore_index)
    return ids, labels



def messages_from_jsonl(path: str) -> List[List[Message]]:
    """Load conversations from a .jsonl file.

    Accepts either format:
      {"messages": [{"role": "user", "content": "..."}, ...]}
      {"instruction": "...", "input": "...", "output": "..."}   (Alpaca style)
    """
    convos: List[List[Message]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: bad JSON ({e})") from e
            if "messages" in obj:
                msgs = [Message(m["role"], m["content"]) for m in obj["messages"]]
                system = obj.get("system")
                if system and not any(m.role == "system" for m in msgs):
                    msgs.insert(0, Message("system", system))
                convos.append(msgs)
            elif "output" in obj:
                prompt = obj["instruction"]
                if obj.get("input"):
                    prompt = f"{prompt}\n{obj['input']}"
                convos.append([Message("user", prompt), Message("assistant", obj["output"])])
            else:
                raise ValueError(f"{path}:{line_no}: need 'messages' or 'output', got {list(obj)[:5]}")
    return convos


def messages_to_jsonl(convos: Sequence[Sequence[Message]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for convo in convos:
            f.write(json.dumps(
                {"messages": [{"role": m.role, "content": m.content} for m in convo]},
                ensure_ascii=False,
            ) + "\n")
