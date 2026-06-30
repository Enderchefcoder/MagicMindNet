# Deep review pass 35 — 2026-05-31

## Scope

Exception surface, dataset corpus getters, merge/loss/export behavior, Linux lint script, GHA smoke consolidation, export parent-dir fix.

## Changes

- Tests: `test_public_exceptions.py`, `test_dataset_corpus_getters.py`, `test_spin_callable.py`, `test_merge_preserves_parameters.py`, `test_chatbot_compute_loss_finite.py`, `test_export_writes_file.py`
- Fix: `mmn-io` `write_file_create_parents` for safetensors/bin/classifier export; Rust test `export_safetensors_creates_parent_directory`
- `scripts/lint.sh`; GHA single `smoke_examples.sh` step; `AGENTS.md` Linux gate docs

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 51
pytest: 133 passed
ruff: clean
```

## TDD

- `test_export_writes_file` failed (os error 3) → export mkdir fix → green

## Merge-ready

YES
