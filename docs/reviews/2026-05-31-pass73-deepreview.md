# Deep review pass 73 — 2026-05-31

## Scope

`mmn-py` split plan steps 4a + 5a: first dataset and model slices.

## Changes

- **`crates/mmn-py/src/datasets/qa.rs`** — `PyDatasetQA` with `pub(crate) inner`
- **`crates/mmn-py/src/datasets/mod.rs`** — re-exports
- **`crates/mmn-py/src/models/diffusion.rs`** — `PyDiffusion` + `smoke_step`
- **`crates/mmn-py/src/models/mod.rs`** — re-exports
- **`lib.rs`** — registers `mod datasets; mod models;`, drops inline QA/Diffusion

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 172
pytest: 422 passed
ruff: clean
```

## Merge-ready

YES

## Next (pass 74)

- `datasets/corpus.rs` (`PyDatasetCorpus`)
