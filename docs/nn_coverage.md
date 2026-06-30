# nn coverage (`mmn-nn`)

Regression coverage for transformer building blocks in `crates/mmn-nn/src/lib.rs`.

## LayerNorm + GELU

| Behavior | Test module |
|----------|-------------|
| Row mean ≈ 0 after LN | `layernorm_tests::layernorm_row_mean_near_zero` |
| LN backward vs finite diff | `layernorm_tests::layernorm_backward_matches_finite_diff` |
| LN γ/β backward vs finite diff | `layernorm_tests::layernorm_backward_gamma_beta_finite_diff` |
| Block FFN residual `out = x2 + ffn` | `attention_tests::transformer_block_uses_ffn_residual` |
| Block backward input vs finite diff | `attention_tests::transformer_block_backward_input_matches_finite_diff` |
| GELU backward vs finite diff | `layernorm_tests::gelu_backward_matches_finite_diff` |

See [layernorm_coverage.md](layernorm_coverage.md).

## Attention + TransformerBlock

| Behavior | Test |
|----------|------|
| Attn output shape `[seq, d_model]` | `attention_tests::attn_forward_preserves_batch_and_d_model` |
| Scaled dot-product uses K/V (not Q-only) | `attention_tests::attn_forward_uses_k_and_v_projections` |
| seq=1 softmax → identity on V | `attention_tests::scaled_dot_product_attention_weights_sum_to_one` |
| SDP backward finite diff (bidirectional) | `attention_tests::scaled_dot_product_attention_backward_bidirectional_finite_diff` |
| Causal mask blocks future keys | `attention_tests::scaled_dot_product_causal_masks_future_keys` |
| Causal SDP backward finite diff | `attention_tests::scaled_dot_product_causal_backward_matches_finite_diff` |
| Sinusoidal PE differs by position | `position_encoding_tests::sinusoidal_pe_differs_by_position` |
| Chatbot causal attn default | `chatbot_tests::chatbot_uses_causal_attention_by_default` |
| Block forward changes activations | `attention_tests::transformer_block_forward_changes_hidden` |
| FFN cache tuple shapes | `attention_tests::transformer_block_forward_with_cache_shapes` |

See [attention_coverage.md](attention_coverage.md) (including **Roadmap** — scaled dot-product backward milestones).

## Running

```powershell
cargo test -p mmn-nn
```
