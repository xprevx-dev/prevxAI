#!/usr/bin/env bash
# Push this project to your GitHub repo.
#
# Arena has no GitHub connector, so this is the handoff step: the repo is
# already committed locally, you just need to point it at your remote.
# Run from the project root:
#
#     ./scripts/push.sh https://github.com/YOUR_USER/YOUR_REPO.git
#
# If the remote already has content you want to keep, use --merge instead.
set -euo pipefail

REMOTE_URL="${1:-}"
MODE="${2:-}"

if [[ -z "$REMOTE_URL" ]]; then
  echo "usage: ./scripts/push.sh <https://github.com/user/repo.git> [--merge]" >&2
  exit 1
fi

cd "$(dirname "$0")/.."

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "[push] initialising git repo"
  git init -q
  git add -A
  git commit -q -m "myai: from-scratch transformer LLM, tokenizer, training, chat, server"
fi

BRANCH="$(git symbolic-ref --short HEAD 2>/dev/null || echo main)"
echo "[push] branch: $BRANCH"

if git remote get-url origin >/dev/null 2>&1; then
  echo "[push] origin already set to $(git remote get-url origin); updating it"
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
fi

if [[ "$MODE" == "--merge" ]]; then
  echo "[push] fetching and merging remote history first"
  git fetch origin
  git merge --allow-unrelated-histories "origin/$BRANCH" || {
    echo "[push] merge needs your attention; resolve conflicts then run:" >&2
    echo "       git push -u origin $BRANCH" >&2
    exit 1
  }
fi

echo "[push] pushing to $REMOTE_URL"
git push -u origin "$BRANCH"
echo "[push] done."
