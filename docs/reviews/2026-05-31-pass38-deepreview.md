# Deep review pass 38 — 2026-05-31

## Scope

Merge vocab regression coverage, autoset sub-10B budget, quantize getter stability, classifier nested import, DataMismatchError messaging, API docs.

## Changes

- Tests: `test_merge_vocab_mismatch`, `test_autoset_sub_10b`, `test_quantize_preserves_getters`, `test_classifier_nested_import_roundtrip`, `test_data_mismatch_error_message`, `test_merge_classifier_callable`
- Docs: `docs/API.md` — `merge()` requires matching `vocab_size`, `n_layer`, `d_model`
- README pytest count **154**

## TDD

- New tests written first; all passed on first green run (behavior already implemented in pass 37 for vocab guard)

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 54
pytest: 154 passed
ruff: clean
```

## Merge-ready

YES
