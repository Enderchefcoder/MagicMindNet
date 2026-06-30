# Deep review pass 49 — 2026-05-31

## Scope

Complete layernorm merge parity, remaining missing-block import keys, and attn.q/ffn2 quantize coverage.

## Changes

- Rust tests (+9): missing attn.k/ln1.gamma import; merge ln1.beta/ln2.gamma/ln2.beta; int8 attn.q/ffn2; int4 attn.k/q
- Python tests (+9): same mirrors; refactored missing-block tests with `_assert_import_fails_missing_tensor`

## TDD

- Regression tests only; no production code changes

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 124
pytest: 233 passed
ruff: clean
```

## Merge-ready

YES
