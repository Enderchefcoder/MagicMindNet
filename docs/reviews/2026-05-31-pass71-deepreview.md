# Deep review pass 71 — 2026-05-31

## Scope

`mmn-py` module split plan step 1: extract errors.

## Changes

- **`crates/mmn-py/src/errors.rs`** — `CPUError` … `ModelMismatchError` + `mmn_err_to_py`
- **`crates/mmn-py/src/lib.rs`** — `mod errors`; imports from errors module (`lib.rs` ~753 lines, was ~778)
- **`docs/mmn_py_split_plan.md`** — step 1 marked done

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 172
pytest: 422 passed (test_public_exceptions unchanged)
ruff: clean
```

## Merge-ready

YES

## Next (pass 72)

- Split plan step 2: `train_config.rs` (`PyTrainConfig`)
- Optional: `resource.rs` (`limit_resources`, `limit_percent`) as quick second slice
