# Deep review pass 39 — 2026-05-31

## Scope

Checkpoint import strictness (no silent partial load), corrupt/missing tensor paths, classifier IO error coverage, merge edge cases, API docs.

## Changes

- **Fix:** `mmn-io` `require_tensor_entry` — import fails on missing embed/lm_head/block tensors or classifier backbone/head; meta requires `n_layer`/`d_model` (no silent defaults)
- Rust tests (+7): int4 quantize, d_model merge, vision OR merge, missing embed, incomplete meta, tensor length mismatch, missing labels meta
- Python tests (+8): classifier import errors, d_model merge, vision merge, int4 weight change, classifier seed import, SPIN CPU smoke
- Docs: `docs/API.md` classifier factories, classifier IO, int4/int8, merge vision OR, import validation

## TDD

- Import strictness: Rust tests written with fix (1 correctness red→green cycle on missing-embed path verified manually before merge)

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 61
pytest: 162 passed
ruff: clean
```

## Merge-ready

YES
