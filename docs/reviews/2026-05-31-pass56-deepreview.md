# Deep review pass 56 — 2026-05-31

## Scope

Classifier edge cases, shared test harness, example smoke gaps, hybrid training regression.

## Changes

- `tests/conftest.py` — shared `project_python`, `run_example` fixture for subprocess example smoke
- `tests/test_classifier_edge_cases_py.py` — single label, unknown tags in mean loss, empty predict, hybrid `TrainClassifier`
- `tests/test_examples_scripts_py.py` — uses conftest fixture; adds `classification_benchmark.py` smoke
- `scripts/smoke_examples.ps1` + `.sh` — `classification_benchmark.py`
- `docs/classifier_coverage.md` — classifier regression matrix
- Rust: `mean_classification_loss_skips_unknown_tags`, `train_step_hybrid_updates_ffn2`

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 152
pytest: 386 passed
ruff: clean
```

## Merge-ready

YES
