#!/usr/bin/env bash
# Assemble the Hugging Face Space (Docker SDK) from this repo and upload it.
# Usage: HF_TOKEN=... scripts/sync_space.sh [space_id]
#
# Run by hand, not by CI. Hugging Face answers 402 Payment Required on
# repos/create for a Docker Space on a free account, so the Space does not exist
# and the GitHub workflow that called this on every push to main only turned main
# red (removed in 2026-11, see git history for sync-space.yml).
set -euo pipefail
SPACE="${1:-Halleck45/OpenPronounce}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set: a token with write access to $SPACE is required." >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cp -r "$ROOT/openpronounce" "$ROOT/templates" "$ROOT/static" "$TMP/"
cp "$ROOT/server.py" "$ROOT/pyproject.toml" "$ROOT/Dockerfile" "$ROOT/.dockerignore" "$TMP/"
cp "$ROOT/docs/space-README.md" "$TMP/README.md"
find "$TMP" -name __pycache__ -type d -exec rm -rf {} +

exists="$(python3 - "$SPACE" <<'PY'
import sys
from huggingface_hub import HfApi
print("yes" if HfApi().repo_exists(sys.argv[1], repo_type="space") else "no")
PY
)"

if [ "$exists" = "no" ]; then
  echo "Space $SPACE does not exist, creating it with the Docker SDK."
  if ! hf repo create "$SPACE" --repo-type space --space-sdk docker; then
    echo "Creation refused. A 402 means the account has no PRO subscription:" >&2
    echo "Hugging Face only hosts Docker Spaces on a paid plan." >&2
    exit 1
  fi
fi

hf upload "$SPACE" "$TMP" . --repo-type space --delete "*" --commit-message "Sync from GitHub"
echo "https://huggingface.co/spaces/$SPACE"
