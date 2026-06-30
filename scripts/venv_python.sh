#!/usr/bin/env bash
# Print repo Python executable (prefer .venv when present).
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -x "$ROOT/.venv/bin/python" ]; then
  echo "$ROOT/.venv/bin/python"
elif [ -x "$ROOT/.venv/Scripts/python.exe" ]; then
  echo "$ROOT/.venv/Scripts/python.exe"
else
  echo "python"
fi
