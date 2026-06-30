# Deep review pass 104 — 2026-05-31

## Scope

Train-then-export learned PE roundtrip example.

## Changes

- **Example** — `learned_pos_embed_roundtrip.py --train` trains before export/import parity check
- **pytest** — `test_learned_pos_embed_roundtrip_train_example_runs`
- **Smoke** — `smoke_examples.ps1` train variant
- **Docs** — `examples/README.md`, `examples_coverage.md`, `position_encoding_coverage.md`

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
pytest: 465 (+1)
learned_pos_embed_roundtrip --train: loss 4.92 → 4.92
```

## Merge-ready

YES

## Next (pass 105)

- Rust `mmn-train` + `mmn-io` dev-dep integration test (train → export → import loss)
- `training_coverage.md` roundtrip `--train` row
