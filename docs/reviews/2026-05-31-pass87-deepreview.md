# Deep review pass 87 — 2026-05-31

## Scope

Merge/quantize regression coverage for learned position embeddings; Train() pytest.

## Changes

- **Rust IO tests** — `merge_models_averages_pos_embed`, `merge_rejects_pos_embed_settings_mismatch`, `quantize_int8/int4_changes_pos_embed_weights`
- **Python** — `test_merge_learned_pos_embed_averages_weights`, `test_merge_rejects_learned_vs_sinusoidal_pos_embed`, `test_train_changes_learned_pos_embed`
- **`docs/position_encoding_coverage.md`** — test matrix rows for merge/quantize/train

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 193 (+4)
pytest: 437 (+3)
ruff: clean
```

## Merge-ready

YES

## Next (pass 88)

- RL/SPIN should keep learned `pos_embed` frozen (mirror attn/LN policy)
- `parameters()` count includes learned PE weights
- Import `pos_embed` shape mismatch test
