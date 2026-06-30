#!/usr/bin/env bash
# Merge gate: full local CI + test counts (repo root).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$(bash "$(dirname "$0")/venv_python.sh")"

if [ -f scripts/ci_local.sh ]; then
  bash scripts/ci_local.sh
elif [ -f scripts/ci_local.ps1 ]; then
  echo "On Windows use: .\\scripts\\verify_gate.ps1"
  exit 1
else
  cargo test --workspace
  maturin develop --release -m crates/mmn-py/Cargo.toml
  "$PY" -m pytest -q
  if command -v ruff >/dev/null 2>&1; then
    ruff check python tests examples
  fi
  "$PY" examples/quickstart.py
fi

if [ -f scripts/count_tests.sh ]; then
  bash scripts/count_tests.sh
else
  "$PY" scripts/count_tests.py 2>/dev/null || true
fi

echo "verify_gate: OK"
