# Deep review pass 43 — 2026-05-31

## Scope

Expand IO/quantize test coverage for pass-42 validation paths: ffn/ln/lm_head shapes, missing meta/tensors, first-path import, quantize parity.

## Changes

- Rust tests (+8): missing `d_model` meta, ffn/ln/lm_head shape mismatch, missing block ffn, bin invalid/empty JSON, classifier int4 quantize
- Python tests (+13): mirror import strict paths, first-path-only import, n_layer merge mismatch, int4 block/head quantize, classifier head shape, bin corrupt files
- Docs: CHANGELOG pass 43 entry; README test counts

## TDD

- All pass-43 items are regression tests for existing strict import/quantize behavior (0 production changes; tests written against current code)

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 86
pytest: 197 passed
ruff: clean
```

## Merge-ready

YES
