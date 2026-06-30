#!/usr/bin/env bash
# Ruff lint — run from repo root with dev deps installed.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v ruff >/dev/null 2>&1; then
  echo "ruff not on PATH; install dev extras: pip install -e '.[dev]'"
  exit 1
fi

ruff check python tests examples
echo "lint: OK"
