# Deep review pass 102 — 2026-05-31

## Scope

Checkpoint roundtrip after training with learned `pos_embed`.

## Changes

- **pytest** — `test_export_import_preserves_learned_pos_embed_after_train` in `test_export_loss.py`
- **Docs** — `checkpoint_coverage.md`, `position_encoding_coverage.md`, `training.md`

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
pytest: 464 (+1)
```

## Merge-ready

YES

## Next (pass 103)

- `API.md` / `examples_coverage.md` flag docs sync
- Optional: Rust train→export learned PE integration test in `mmn-train`
