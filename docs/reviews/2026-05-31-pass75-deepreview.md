# Deep review pass 75 — 2026-05-31

## Scope

`mmn-py` split plan step 4 complete: remaining dataset pyclasses.

## Changes

- **`crates/mmn-py/src/datasets/classification.rs`** — `PyDatasetClassification` + `unique_labels`
- **`crates/mmn-py/src/datasets/image.rs`** — `PyDatasetImageGen`, `PyDatasetImageEdit`
- **`datasets/mod.rs`** — full re-export surface for all five dataset types
- **`lib.rs`** — ~421 lines (was ~529); datasets module fully extracted

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 172
pytest: 422 passed
ruff: clean
```

## Merge-ready

YES

## Next (pass 76)

- Split plan step 5b: `models/chatbot.rs` (`PyChatbot` — largest model pyclass)
