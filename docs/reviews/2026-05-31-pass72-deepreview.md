# Deep review pass 72 — 2026-05-31

## Scope

`mmn-py` split plan steps 2–3: `train_config.rs` and `resource.rs`.

## Changes

- **`crates/mmn-py/src/train_config.rs`** — `PyTrainConfig` + `to_train_config()`
- **`crates/mmn-py/src/resource.rs`** — `limit_resources`, `limit_percent`
- **`lib.rs`** — ~698 lines (was ~753); registers modules from split files

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 172
pytest: 422 passed
ruff: clean
```

## Merge-ready

YES

## Next (pass 73)

- Split plan step 4a: `datasets/qa.rs` (`PyDatasetQA`) + `datasets/mod.rs`
- Or `models/diffusion.rs` as smallest model pyclass slice
