# Deep review pass 79 — 2026-05-31

## Scope

Post–split `mmn-py` documentation and binding regression smoke.

## Changes

- **`docs/mmn_py_coverage.md`** — Rust module → Python symbol → pytest matrix
- **`tests/test_mmn_py_bindings_py.py`** — 7 tests: `_native` exports, pyclass names, construct + repr, Diffusion smoke, exception type, chatbot export/import roundtrip
- **`docs/testing.md`**, **`CONTRIBUTING.md`** — index links; split plan marked complete
- **`docs/mmn_py_split_plan.md`** — header updated to “complete”

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 172
pytest: 429 passed
ruff: clean
```

## Merge-ready

YES

## Next (pass 80)

- Attention backward milestone (design in `attention_coverage.md`) or initial git/PR bootstrap (NEEDS HUMAN)
