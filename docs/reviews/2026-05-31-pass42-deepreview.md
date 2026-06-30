# Deep review pass 42 — 2026-05-31

## Scope

Block-level safetensors shape validation, classifier corrupt-file parity, corpus fixed batch getter, int4 quantize weight change.

## Changes

- **Fix:** `import_block_tensors` validates all block weights vs `d_model` and `ffn_dim` after load
- Rust tests (+5): block shape mismatch, n_layer meta mismatch, classifier invalid JSON/empty/backbone shape
- Python tests (+7): mirror block/classifier strict paths, fixed corpus batch, int4 quantize backbone diff
- Docs: `API.md` first-path-only import + block validation; `checkpoints.md` block tensor shapes; `magicmindnet-io` agent

## TDD

- `import_rejects_block_tensor_shape_mismatch` — corrupt `blocks.0.attn.q` shape in JSON, then add inline validation (1 red→green)

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 78
pytest: 184 passed
ruff: clean
```

## Merge-ready

YES
