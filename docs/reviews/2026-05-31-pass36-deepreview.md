# Deep review pass 36 — 2026-05-31

## Scope

Training API callables, classifier/chatbot seed & vision roundtrip, mean-loss finiteness, classifier nested export, Rust IO test.

## Changes

- Tests: `test_train_callable.py`, `test_rl_callable.py`, `test_merge_callable.py`, `test_classifier_init_seed_getter.py`, `test_safetensors_vision_roundtrip.py`, `test_chatbot_compute_mean_loss_finite.py`, `test_export_classifier_nested_path.py`
- Rust: `export_classifier_creates_parent_directory` in `mmn-io`
- Docs: `testing.md`, `CHANGELOG`, `README` counts

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 52
pytest: 141 passed
ruff: clean
```

## Merge-ready

YES
