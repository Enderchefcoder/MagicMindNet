# Deep review pass 27 — 2026-05-31

## Changes

- `mmn-bin-v1` on export/import bin; reject full checkpoints via `import_bin`
- Tests: bin IO, unknown classifier label, autoset budget
- Examples smoke: `eval_mean_loss.py qa`

## Verification

- `cargo test --workspace`: 50 Rust tests, 0 failed
- `pytest -q`: 90 passed
- `ruff`: clean
- `.\scripts\ci_local.ps1`: exit 0 (`mean QA loss` in smoke)
