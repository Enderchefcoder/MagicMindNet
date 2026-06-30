# Deep review pass 91 — 2026-05-31

## Scope

Corpus LM training with learned position embeddings; bin PE docs; training guide accuracy fix.

## Changes

- **Rust** — `train_corpus_updates_learned_pos_embed`; `bin_vision_and_learned_pos_embed_roundtrip_preserves_meta`
- **Python** — `test_train_corpus_learned_pos_embed_reduces_mean_loss`, `test_bin_roundtrip_vision_and_learned_pos_embed_meta`
- **Docs** — `vision_coverage.md` bin PE row; `position_encoding_coverage.md` corpus rows; `training.md` attn/LN/PE scope + corpus PE note

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 199 (+2)
pytest: 445 (+2)
ruff: clean
```

## Merge-ready

YES

## Next (pass 92)

- Post-import `Train(DatasetCorpus)` with learned PE
- `training_coverage.md` position-encoding rows
