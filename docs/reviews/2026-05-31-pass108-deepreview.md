# Deep review pass 108 — 2026-05-31

## Scope

Post-train quantize (int4) and merge trained learned-PE checkpoints.

## Changes

- **pytest** — `test_quantize_int4_learned_pos_embed_after_train_within_tolerance`
- **pytest** — `test_merge_trained_learned_pos_embed_averages_weights`
- **Docs** — `checkpoint_coverage.md`, `quantize_coverage.md`

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
pytest: 468 (+2)
```

## Merge-ready

YES

## Next (pass 109)

- Rust quantize-after-train learned PE (optional)
- `quickstart.py` learned PE comment or `--learned-pe` flag
