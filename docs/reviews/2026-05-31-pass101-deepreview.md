# Deep review pass 101 — 2026-05-31

## Scope

QA benchmark with learned position embeddings.

## Changes

- **Example** — `benchmark_train.py --learned-pe`
- **pytest** — `test_benchmark_train_learned_pe_example_runs`
- **Smoke** — `smoke_examples.ps1` learned-pe variant
- **Docs** — `examples/README.md`, `examples_coverage.md`, `position_encoding_coverage.md`, `training_coverage.md`

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
pytest: 463 (+1)
benchmark_train --learned-pe: loss 5.72 → 0.72
```

## Merge-ready

YES
