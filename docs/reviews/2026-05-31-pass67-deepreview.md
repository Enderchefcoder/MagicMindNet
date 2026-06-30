# Deep review pass 67 — 2026-05-31

## Scope

SPIN frozen LN pytest, Linux verify_gate venv fallback, limitations roadmap table.

## Changes

- **`tests/test_train_rl_spin_py.py`** — `test_spin_does_not_change_ln1_gamma`
- **`scripts/verify_gate.sh`** — fallback CI path uses `venv_python.sh` for pytest/quickstart
- **`docs/limitations.md`** — RL/SPIN frozen note + post-alpha roadmap table
- **Coverage docs** — `attention_coverage`, `layernorm_coverage`, `training_coverage` SPIN LN row

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 172
pytest: 418 passed
ruff: clean
```

## Merge-ready

YES

## Next (pass 68)

- Add `limitations.md` to CONTRIBUTING coverage table
- Consolidate frozen-param tests helper in `tests/conftest.py` (optional DRY)
- `mmn-nn` attention backward design note in `attention_coverage.md` roadmap section
