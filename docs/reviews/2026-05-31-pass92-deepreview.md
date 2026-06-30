# Deep review pass 92 — 2026-05-31

## Scope

Post-import training with learned position embeddings; training coverage matrix PE rows.

## Changes

- **`test_train_corpus_after_import_learned_pos_embed_reduces_loss`** — import safetensors, corpus `Train()`, loss + `pos_embed` update
- **`test_train_after_import_learned_pos_embed_reduces_loss`** — same for QA dataset
- **`training_coverage.md`** — learned PE train/corpus/post-import/RL/SPIN rows + doc link
- **`position_encoding_coverage.md`** — post-import test row

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 199 (unchanged)
pytest: 447 (+2)
ruff: clean
```

## Merge-ready

YES

## Next (pass 93)

- Quantize learned PE preserves forward loss within tolerance
- `checkpoint_coverage.md` learned `pos_embed` section
