# Deep review pass 80 — 2026-05-31

## Scope

Attention roadmap milestone 1: scaled dot-product **forward** in `mmn-nn`.

## Changes

- **`scaled_dot_product_attention(q, k, v, n_heads)`** — per-head QKᵀ/√head_dim, row softmax, weighted V merge
- **`MultiHeadAttention::forward`** — uses Q/K/V projections + scaled dot-product (replaces Q-only shortcut)
- **Tests** — `attn_forward_uses_k_and_v_projections`, `scaled_dot_product_attention_weights_sum_to_one`, `scaled_dot_product_uniform_queries_blend_values`
- **Docs** — `attention_coverage.md`, `nn_coverage.md`, `limitations.md` (forward done; backward still alpha gap)

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 174 (+2)
pytest: 429 passed
ruff: clean
```

Frozen attn/LN tests unchanged — backward not wired.

## Merge-ready

YES

## Next (pass 81)

- Attention softmax backward + `train_step_lm` integration (large); or LayerNorm backward sketch
