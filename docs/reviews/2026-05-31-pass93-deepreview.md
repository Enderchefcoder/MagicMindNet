# Deep review pass 93 — 2026-05-31

## Scope

Quantize loss tolerance for learned position embeddings; checkpoint coverage documentation.

## Changes

- **Rust** — `quantize_int8/int4_learned_pos_embed_preserves_forward_loss_within_tolerance`
- **Python** — `test_learned_pos_embed_quantize_py.py` (int8/int4 mean-loss drift, meta preserved on re-export)
- **Docs** — `checkpoint_coverage.md` learned `pos_embed` section; `quantize_coverage.md` + `position_encoding_coverage.md` rows

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 201 (+2)
pytest: 450 (+3)
ruff: clean
```

## Merge-ready

YES

## Next (pass 94)

- Add `pos_embed` to `test_io_checkpoint_matrix_py.py` parametric coverage (learned PE models)
