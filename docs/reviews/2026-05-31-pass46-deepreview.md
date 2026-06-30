# Deep review pass 46 — 2026-05-31

## Scope

Close remaining checkpoint-strict and merge/quantize parity gaps: attn.v/out import shape tests, Python merge lm_head/backbone averaging, int8 attn.v quantize.

## Changes

- Rust tests (+3): `import_rejects_attn_v_shape_mismatch`, `import_rejects_attn_out_shape_mismatch`, `quantize_int8_changes_block_attn_v_weights`
- Python tests (+4): attn.v/out import strict, merge chatbot lm_head average, merge classifier backbone average

## TDD

- Regression tests only; no production code changes

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 100
pytest: 209 passed
ruff: clean
```

## Merge-ready

YES
