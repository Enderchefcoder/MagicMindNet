# Deep review pass 64 — 2026-05-31

## Scope

Testing docs index, agent coverage pointers, Python frozen attn/LN regression.

## Changes

- **`docs/testing.md`** — full `docs/*_coverage.md` index + quick Rust/Python map
- **`AGENTS.md`** — coverage matrices section linking `testing.md` and `CONTRIBUTING.md`
- **`tests/test_train_frozen_attn_ln_py.py`** — attn.q and ln1.gamma unchanged after `Train`; ffn2 positive control
- **`docs/training_coverage.md`** — frozen attn/LN rows + Python mirror
- **`README.md`** — pytest count 414

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 172
pytest: 414 passed
ruff: clean
```

## Merge-ready

YES

## Next (pass 65)

- Fix `count_tests.ps1` quickstart `ModuleNotFoundError` (venv path)
- `CONTRIBUTING.md` link to `docs/testing.md` as master index
- Optional: `test_train_frozen_attn_ln_py` RL mirror (attn still frozen under RL? — verify behavior first)
- Example smoke for `classification.py` if low-cost
