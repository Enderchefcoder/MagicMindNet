# Deep review pass 48 — 2026-05-31

## Scope

Close FFN merge/quantize parity and missing block tensor import strictness gaps.

## Changes

- Rust tests (+8): missing attn.q/ffn2 import; merge ffn/ffn2/ln1.gamma; int8 attn.k; int4 attn.out/ffn2
- Python tests (+8): missing attn.q/ffn2 import; merge ffn/ffn2/ln1.gamma; int8 attn.k; int4 attn.out/ffn2

## TDD

- Regression tests only; no production code changes

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 115
pytest: 224 passed
ruff: clean
```

## Merge-ready

YES
