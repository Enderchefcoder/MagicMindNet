# Deep review pass 74 — 2026-05-31

## Scope

`mmn-py` split plan step 4b: `PyDatasetCorpus`.

## Changes

- **`crates/mmn-py/src/datasets/corpus.rs`** — `PyDatasetCorpus` (load, getters, `corpus_batch_size`, `__repr__`)
- **`datasets/mod.rs`** — re-export `PyDatasetCorpus`
- **`lib.rs`** — ~529 lines; `Train` / `compute_mean_loss` downcast via imported type

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 172
pytest: 422 passed
ruff: clean
```

## Merge-ready

YES

## Next (pass 75)

- Complete `datasets/`: `classification.rs` + `image.rs`
