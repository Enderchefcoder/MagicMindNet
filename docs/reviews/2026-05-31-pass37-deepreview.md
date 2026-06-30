# Deep review pass 37 — 2026-05-31

## Scope

Autoset sub-1B, classifier loss finiteness, bin nested export, IO alias equivalence, merge vocab guard bugfix.

## Changes

- Tests: autoset sub-1B/seed, classifier compute_loss/mean_loss finite, bin nested export, IO aliases, ModelMismatchError on vocab merge
- **Fix:** `merge_models` validates `vocab_size` before tensor average (was panic); Rust test `merge_rejects_vocab_mismatch`
- Rust: `export_bin_creates_parent_directory`; GHA `count_tests.sh` on all OS

## TDD

- `test_model_mismatch_error` failed with `PanicException` → vocab guard → green (1 red→green cycle)

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 54
pytest: 148 passed
```

## Merge-ready

YES
