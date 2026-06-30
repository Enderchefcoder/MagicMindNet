# Deep review pass 65 — 2026-05-31

## Scope

CI venv Python resolution, classification example smoke, RL frozen-attn pytest.

## Changes

- **`scripts/venv_python.ps1` / `venv_python.sh`** — resolve `.venv` Python for all gate scripts
- **`ci_local.ps1` / `.sh`, `smoke_examples.ps1` / `.sh`, `count_tests.ps1` / `.sh`** — use venv Python; tighter `Invoke-Step` exit-code check
- **`smoke_examples`** — `classification.py` added (full example parity with pytest)
- **`tests/test_train_rl_spin_py.py`** — `test_rl_does_not_change_attn_q_weights`
- **Docs** — `examples_coverage.md`, `training_coverage.md`, `classifier_coverage.md`, `AGENTS.md`

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK (quickstart + full smoke pass without venv activation)
Rust #[test]: 172
pytest: 415 passed
ruff: clean
```

## Merge-ready

YES

## Next (pass 66)

- Document `venv_python` in `docs/testing.md` / `CONTRIBUTING.md`
- Optional: `test_rl_does_not_change_ln1_gamma` mirror
- Continue attention backward roadmap notes or mmn-py split planning
