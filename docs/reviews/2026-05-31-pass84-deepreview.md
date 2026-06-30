# Deep review pass 84 — 2026-05-31

## Scope

Causal attention for LM, test module rename, vision cross-attn roadmap.

## Changes

- **`scaled_dot_product_attention*`** — `causal: bool` param; `MultiHeadAttention.causal` defaults **true**
- **Tests** — `scaled_dot_product_causal_masks_future_keys`, causal + bidirectional backward finite-diff
- **`test_train_block_params_py.py`** — replaces misleading `test_train_frozen_attn_ln_py.py`; +`test_train_changes_embed_and_lm_head`
- **`docs/vision_coverage.md`** — cross-attention design sketch for `vision=True`

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 181 (+2)
pytest: 430 (+1)
ruff: clean
```

## Merge-ready

YES

## Next (pass 85)

- `MultiHeadAttention.causal` getter in Python / checkpoint meta (optional)
- Position embeddings or RoPE sketch for longer contexts
- Classifier byte-feature encoder improvements
