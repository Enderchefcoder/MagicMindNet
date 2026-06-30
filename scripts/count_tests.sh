#!/usr/bin/env bash
# Print approximate Rust + Python test counts (from repo root).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$(bash "$(dirname "$0")/venv_python.sh")"

rust=0
if command -v rg >/dev/null 2>&1; then
  while IFS= read -r count; do
    rust=$((rust + count))
  done < <(rg -c '#\[test\]' crates --glob '*.rs' 2>/dev/null | cut -d: -f2)
else
  rust=$(grep -r '#\[test\]' crates --include='*.rs' 2>/dev/null | wc -l | tr -d ' ')
fi

py_line=$("$PY" -m pytest --collect-only -q 2>/dev/null | tail -1)
py_tests=0
if [[ "$py_line" =~ ([0-9]+)[[:space:]]+tests?[[:space:]]+collected ]]; then
  py_tests="${BASH_REMATCH[1]}"
fi

echo "Rust #[test] in crates/: $rust"
echo "Python tests (pytest --collect-only): $py_tests"
echo "Run bash scripts/ci_local.sh for the full gate."
