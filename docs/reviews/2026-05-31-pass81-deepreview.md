# Deep review pass 81 — 2026-05-31

## Scope

Attention roadmap milestone 2: softmax backward + `train_step_lm` integration.

## Changes

- **`scaled_dot_product_attention_backward`** + `SdpAttentionCache` (weights from forward)
- **`TransformerBlock::forward_with_cache`** / **`backward_attn_ffn`** — FFN + full attn linear grads (LN still identity in backward)
- **`Chatbot::backward_lm_grads`** — 6 weight grads per block (ffn2, ffn, out, q, k, v)
- **Tests** — finite-diff SDP backward; `train_step_updates_attn_q_weights`; pytest `test_train_changes_attn_q_weights`, `test_spin_changes_attn_q_weights`; RL attn still frozen

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 175 (+1)
pytest: 429 passed
ruff: clean
```

## Merge-ready

YES

## Next (pass 82)

- LayerNorm γ/β backward into `train_step_lm`
