# Deep review pass 45 — 2026-05-31

## Scope

Complete remaining IO strict-path and merge/quantize parity gaps from checkpoint-strict scan.

## Changes

- Rust tests (+6): merge embed/lm_head average, attn.k + ln2.gamma shape, missing lm_head tensor, int8 block ffn quantize
- Python tests (+4): embed merge average, attn.k/ln2/lm_head import strict (`test_import_safetensors_remaining_strict_py.py`)

## TDD

- Regression tests only; no production code changes

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 97
pytest: 205 passed
ruff: clean
```

## Merge-ready

YES
