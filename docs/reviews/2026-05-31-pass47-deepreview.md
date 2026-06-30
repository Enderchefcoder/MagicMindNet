# Deep review pass 47 — 2026-05-31

## Scope

Complete layernorm beta import strictness, full attention merge averaging parity, and block attn quantize coverage.

## Changes

- Rust tests (+7): ln1/ln2 beta shape mismatch; merge attn k/v/out averaging; int8 attn.out + int4 attn.v quantize
- Python tests (+7): ln beta import strict; merge attn k/v/out export averaging; int8 attn.out + int4 attn.v quantize
- Refactor: `_merge_export_first_f32` helper in merge chatbot averages tests

## TDD

- Regression tests only; no production code changes

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 107
pytest: 216 passed
ruff: clean
```

## Merge-ready

YES
