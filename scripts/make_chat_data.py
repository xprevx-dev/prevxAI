#!/usr/bin/env python3
"""Generate a small instruction-tuning set so the base model learns the *shape* of
answering a question. This is a bootstrap: hand-written examples beat these, but
these get a from-scratch model talking in Q&A form instead of only continuing text.

The assistant turns are intentionally short and simple so a tiny model can learn
them. Edit the templates freely — this file is yours.

    python scripts/make_chat_data.py --out data/chat.jsonl --n 800
"""

from __future__ import annotations

import argparse
import json
import random

IDENTITY = [
    ("Who are you?", "I am myai, a small language model trained from scratch."),
    ("What is your name?", "My name is myai."),
    ("How were you made?", "I was written from scratch in PyTorch and trained on simple stories."),
    ("Are you a big model?", "No, I am a very small model. I learn best on simple text."),
    ("What did you train on?", "I trained on a lot of short, simple stories."),
    ("Can you help me?", "Yes, I will try my best to help you."),
]

STORY_OPENERS = [
    "Once upon a time", "One day", "Long ago", "One sunny morning", "One rainy evening",
]
CHARACTERS = ["a little girl named Lily", "a small dog named Max", "a kind boy named Tom",
              "a cat named Bella", "a rabbit named Coco"]
PLACES = ["in a small village", "in a green forest", "by a quiet lake", "in a cozy house",
          "on a big farm"]
ACTIONS = ["found a red balloon", "made a new friend", "baked a sweet cake",
           "planted a tiny seed", "lost a shiny coin", "saw a bright rainbow"]
FEELINGS = ["was very happy", "felt proud", "was excited", "smiled with joy", "felt brave"]
MORALS = ["learned that kindness matters.", "learned to always help others.",
          "learned that sharing brings joy.", "learned to never give up."]


def story() -> str:
    return (f"{random.choice(STORY_OPENERS)}, there was {random.choice(CHARACTERS)} "
            f"who lived {random.choice(PLACES)}. One day, this friend {random.choice(ACTIONS)} "
            f"and {random.choice(FEELINGS)}. In the end, they {random.choice(MORALS)}")


def build(n: int, seed: int) -> list[dict]:
    random.seed(seed)
    out: list[dict] = []
    for i in range(n):
        kind = i % 4
        if kind == 0:
            q, a = random.choice(IDENTITY)
        elif kind == 1:
            q = random.choice(["Tell me a short story.", "Tell me a story.",
                               "Can you tell me a story?"])
            a = story()
        elif kind == 2:
            subj = random.choice(CHARACTERS)
            a = story()
            q = f"Tell me a story about {subj.split(' named ')[-1]}."
            a = a.replace(subj, f"a little friend named {subj.split(' named ')[-1]}", 1)
        else:
            q = random.choice(["What is your favorite story about?",
                               "Do you like stories?", "Why do you tell stories?"])
            a = random.choice([
                "I like stories about kind friends who help each other.",
                "Yes, I love stories. They teach us to be kind and brave.",
                "I tell stories because they make people happy.",
            ])
        out.append({"messages": [{"role": "user", "content": q},
                                 {"role": "assistant", "content": a}]})
    random.shuffle(out)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/chat.jsonl")
    p.add_argument("--n", type=int, default=800)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    convos = build(args.n, args.seed)
    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for c in convos:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[chat] wrote {len(convos)} conversations -> {args.out}")


if __name__ == "__main__":
    main()
