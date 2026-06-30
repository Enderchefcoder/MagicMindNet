# Deep review pass 66 — 2026-05-31

## Scope

RL/SPIN frozen-parameter pytest mirrors, coverage doc polish, CONTRIBUTING venv notes.

## Changes

- **`tests/test_train_rl_spin_py.py`** — `test_rl_does_not_change_ln1_gamma`, `test_spin_does_not_change_attn_q_weights`
- **`docs/attention_coverage.md`** — RL/SPIN frozen attn/LN table + Python test links
- **`docs/layernorm_coverage.md`** — RL frozen γ pytest reference
- **`docs/training_coverage.md`** — RL LN + SPIN attn rows
- **`CONTRIBUTING.md`** — `venv_python` scripts; activation optional for gate

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 172
pytest: 417 passed
ruff: clean
```

## Merge-ready

YES

## Next (pass 67)

- `test_spin_does_not_change_ln1_gamma` (optional symmetry)
- Attention backward roadmap stub in `limitations.md` cross-link
- `verify_gate.sh` fallback path: use `venv_python.sh` when ci_local missing
