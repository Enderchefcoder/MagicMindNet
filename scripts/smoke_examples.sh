#!/usr/bin/env bash
# Run non-interactive example scripts (repo root; uses .venv Python when present).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$(bash "$(dirname "$0")/venv_python.sh")"

echo "== examples/quickstart =="
"$PY" examples/quickstart.py

echo "== examples/checkpoint_roundtrip =="
"$PY" examples/checkpoint_roundtrip.py

echo "== examples/classifier_roundtrip =="
"$PY" examples/classifier_roundtrip.py

echo "== examples/classification =="
"$PY" examples/classification.py

echo "== examples/eval_mean_loss qa =="
"$PY" examples/eval_mean_loss.py qa

echo "== examples/eval_mean_loss cls =="
"$PY" examples/eval_mean_loss.py cls

echo "== examples/eval_mean_loss corpus =="
"$PY" examples/eval_mean_loss.py corpus

echo "== examples/benchmark_train =="
"$PY" examples/benchmark_train.py

echo "== examples/rl_spin =="
"$PY" examples/rl_spin.py

echo "== examples/classification_benchmark =="
"$PY" examples/classification_benchmark.py

echo "== examples/corpus_benchmark =="
"$PY" examples/corpus_benchmark.py

echo "== examples/diffusion_smoke =="
"$PY" examples/diffusion_smoke.py

echo "== examples/vision_chatbot =="
"$PY" examples/vision_chatbot.py

echo "Examples smoke: OK"
