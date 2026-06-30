# Deep review pass 40 — 2026-05-31

## Scope

Checkpoint tensor shape validation, classifier meta strictness, safe byte parsing, pytest mirror of Rust import guards, corpus batch_size getter.

## Changes

- **Fix:** `expect_tensor_shape` on chatbot/classifier import (Linear layout `[out, in]`)
- **Fix:** Classifier requires `input_dim` meta and non-empty labels; rejects missing backbone
- **Fix:** `json_byte` safe parsing (no panic on invalid tensor bytes)
- **Feature:** `DatasetCorpus.corpus_batch_size` getter
- Tests: +6 Rust, +9 Python; docs `checkpoints.md` / `API.md`

## TDD

- Shape validation: Rust tests failed with wrong expected dims → corrected to `[vocab, d_model]` / `[128, input_dim]` → green

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 67
pytest: 171 passed
ruff: clean
```

## Merge-ready

YES
