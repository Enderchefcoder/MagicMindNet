# Deep review pass 107 — 2026-05-31

## Scope

Quantize learned PE after training (pytest).

## Changes

- **pytest** — `test_quantize_int8_learned_pos_embed_after_train_within_tolerance`
- **Docs** — `quantize_coverage.md`, `position_encoding_coverage.md`

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
pytest: 466 (+1)
```

## Merge-ready

YES

## Next (pass 108)

- int4 quantize after train symmetry test
- Merge two trained learned-PE checkpoints (pytest)
