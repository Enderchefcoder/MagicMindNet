# Deep review pass 68 — 2026-05-31

## Scope

Checkpoint test helpers in conftest, attention backward design doc, limitations index polish.

## Changes

- **`tests/conftest.py`** — `load_checkpoint_tensors`, `checkpoint_tensor_bytes`, `checkpoint_tensor_first_f32`
- **`test_train_frozen_attn_ln_py.py`**, **`test_train_rl_spin_py.py`** — use shared helpers (DRY)
- **`docs/attention_coverage.md`** — scaled dot-product backward design sketch + milestones
- **`CONTRIBUTING.md`**, **`docs/testing.md`** — `limitations.md` in coverage index

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 172
pytest: 418 passed
ruff: clean
```

## Merge-ready

YES

## Next (pass 69)

- Migrate IO matrix tests to `conftest` first-f32 helper (optional DRY)
- `nn_coverage.md` link to attention backward milestones
- Small pytest for `conftest.project_python` parity with `venv_python.ps1`
