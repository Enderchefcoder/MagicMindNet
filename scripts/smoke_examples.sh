#!/usr/bin/env bash
# Run non-interactive example scripts (repo root; uses .venv Python when present).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$(bash "$(dirname "$0")/venv_python.sh")"

echo "== examples/quickstart =="
"$PY" examples/quickstart.py

echo "== examples/checkpoint_roundtrip =="
"$PY" examples/checkpoint_roundtrip.py

echo "== examples/hf_safetensors_roundtrip =="
"$PY" examples/hf_safetensors_roundtrip.py

echo "== examples/rope_roundtrip =="
"$PY" examples/rope_roundtrip.py

echo "== examples/rope_roundtrip train =="
"$PY" examples/rope_roundtrip.py --train

echo "== examples/classifier_roundtrip =="
"$PY" examples/classifier_roundtrip.py

echo "== examples/classifier_hf_safetensors_roundtrip =="
"$PY" examples/classifier_hf_safetensors_roundtrip.py

echo "== examples/classification =="
"$PY" examples/classification.py

echo "== examples/eval_mean_loss qa =="
"$PY" examples/eval_mean_loss.py qa

echo "== examples/eval_mean_loss cls =="
"$PY" examples/eval_mean_loss.py cls

echo "== examples/eval_mean_loss corpus =="
"$PY" examples/eval_mean_loss.py corpus

echo "== examples/eval_mean_loss qa rope =="
"$PY" examples/eval_mean_loss.py qa --rope

echo "== examples/benchmark_train =="
"$PY" examples/benchmark_train.py

echo "== examples/benchmark_train bpe =="
"$PY" examples/benchmark_train.py --bpe

echo "== examples/benchmark_train rope =="
"$PY" examples/benchmark_train.py --rope

echo "== examples/generate_reply =="
"$PY" examples/generate_reply.py

echo "== examples/rl_spin =="
"$PY" examples/rl_spin.py

echo "== examples/classification_benchmark =="
"$PY" examples/classification_benchmark.py

echo "== examples/corpus_benchmark =="
"$PY" examples/corpus_benchmark.py

echo "== examples/corpus_benchmark bpe =="
"$PY" examples/corpus_benchmark.py --bpe

echo "== examples/corpus_benchmark rope =="
"$PY" examples/corpus_benchmark.py --rope

echo "== examples/diffusion_smoke =="
"$PY" examples/diffusion_smoke.py

echo "== examples/vision_chatbot =="
"$PY" examples/vision_chatbot.py

echo "Examples smoke: OK"
