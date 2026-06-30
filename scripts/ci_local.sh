#!/usr/bin/env bash
# Local CI mirror — run from repo root (uses .venv Python when present).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$(bash "$(dirname "$0")/venv_python.sh")"

echo "== cargo test =="
cargo test --workspace

echo "== maturin develop =="
maturin develop --release -m crates/mmn-py/Cargo.toml

echo "== pytest =="
"$PY" -m pytest -q

echo "== ruff =="
if command -v ruff >/dev/null 2>&1; then
  ruff check python tests examples
else
  echo "skip ruff (pip install -e '.[dev]')"
fi

echo "== quickstart =="
"$PY" examples/quickstart.py

if [ -f scripts/smoke_examples.sh ]; then
  echo "== examples smoke =="
  bash scripts/smoke_examples.sh
fi

echo "CI local: OK"
