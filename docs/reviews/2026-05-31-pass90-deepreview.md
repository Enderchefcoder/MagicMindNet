# Deep review pass 90 — 2026-05-31

## Scope

Learned PE training loss regression; limitations doc alignment.

## Changes

- **`test_learned_pos_embed_compute_mean_loss_decreases_after_train`** — `Train()` reduces mean QA loss with learned `pos_embed`
- **`limitations.md`** — sinusoidal default vs learned PE; `bin` stores PE architecture flags

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 197 (unchanged)
pytest: 443 (+1)
ruff: clean
```

## Merge-ready

YES

## Next (pass 91)

- `Train(DatasetCorpus)` with learned PE smoke test
- `vision_coverage.md` bin PE meta row
