# Run non-interactive example scripts (repo root; uses .venv Python when present).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$PY = & "$PSScriptRoot\venv_python.ps1"

Write-Host "== examples/quickstart =="
& $PY examples/quickstart.py

Write-Host "== examples/checkpoint_roundtrip =="
& $PY examples/checkpoint_roundtrip.py

Write-Host "== examples/learned_pos_embed_roundtrip =="
& $PY examples/learned_pos_embed_roundtrip.py

Write-Host "== examples/learned_pos_embed_roundtrip train =="
& $PY examples/learned_pos_embed_roundtrip.py --train

Write-Host "== examples/classifier_roundtrip =="
& $PY examples/classifier_roundtrip.py

Write-Host "== examples/classification =="
& $PY examples/classification.py

Write-Host "== examples/eval_mean_loss qa =="
& $PY examples/eval_mean_loss.py qa

Write-Host "== examples/eval_mean_loss cls =="
& $PY examples/eval_mean_loss.py cls

Write-Host "== examples/eval_mean_loss corpus =="
& $PY examples/eval_mean_loss.py corpus

Write-Host "== examples/benchmark_train =="
& $PY examples/benchmark_train.py

Write-Host "== examples/benchmark_train learned-pe =="
& $PY examples/benchmark_train.py --learned-pe

Write-Host "== examples/benchmark_train bpe =="
& $PY examples/benchmark_train.py --bpe

Write-Host "== examples/bpe_roundtrip =="
& $PY examples/bpe_roundtrip.py

Write-Host "== examples/bpe_roundtrip train =="
& $PY examples/bpe_roundtrip.py --train

Write-Host "== examples/rl_spin =="
& $PY examples/rl_spin.py

Write-Host "== examples/classification_benchmark =="
& $PY examples/classification_benchmark.py

Write-Host "== examples/corpus_benchmark =="
& $PY examples/corpus_benchmark.py

Write-Host "== examples/corpus_benchmark learned-pe =="
& $PY examples/corpus_benchmark.py --learned-pe

Write-Host "== examples/corpus_benchmark bpe =="
& $PY examples/corpus_benchmark.py --bpe

Write-Host "== examples/diffusion_smoke =="
& $PY examples/diffusion_smoke.py

Write-Host "== examples/vision_chatbot =="
& $PY examples/vision_chatbot.py

Write-Host "Examples smoke: OK"
