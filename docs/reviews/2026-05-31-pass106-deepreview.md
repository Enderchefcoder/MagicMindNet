# Deep review pass 106 — 2026-05-31

## Scope

Corpus LM train → export → import learned PE (Rust).

## Changes

- **Rust** — `train_corpus_learned_pos_embed_export_import_preserves_mean_loss`
- **Docs** — `training_coverage.md` corpus integration row

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 204 (+1)
```

## Merge-ready

YES
