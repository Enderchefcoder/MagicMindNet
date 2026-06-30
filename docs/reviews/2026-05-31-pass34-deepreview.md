# Deep review pass 34 — 2026-05-31

## Scope

Getter/repr/export-surface tests, Linux CI script parity, GHA test-count step.

## Changes

- Tests: `layer_size` vs `n_layer`, `has_vision`, Diffusion `repr`, `__all__` exports, `TrainClassifier` callable
- Scripts: `ci_local.sh`, `count_tests.sh`, `smoke_examples.sh`
- GHA: `bash scripts/count_tests.sh` on Ubuntu
- Docs: `testing.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, README counts

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 50
pytest: 123 passed
ruff: clean
```

## Merge-ready

YES
