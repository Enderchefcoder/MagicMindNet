# Deep review pass 50 — 2026-05-31

## Scope

100% chatbot safetensors IO contract coverage (documented matrix), parametric pytest suite, massive README expansion.

## Changes

- `tests/test_io_checkpoint_matrix_py.py` — parametric missing/shape/merge/quantize for all 12 tensor keys
- `docs/checkpoint_coverage.md` — full regression matrix (layernorm quantize noted as code-path at init)
- Rust (+5): missing attn.v/out, ln1.beta, ln2.gamma/beta import tests
- README.md — architecture, features, API, IO strictness, docs index, project layout
- Updated `docs/checkpoints.md`, `docs/testing.md`, checkpoint-strict subagent

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 129
pytest: 285 passed
ruff: clean
```

## Merge-ready

YES
