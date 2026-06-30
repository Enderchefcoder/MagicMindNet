# LayerNorm coverage (alpha)

Per-block LayerNorm (`ln1`, `ln2`) in `mmn-nn` normalizes each `[batch, d_model]` row, then applies learned γ (gamma) and β (beta).

## Forward

Implementation: `LayerNorm::forward` in `crates/mmn-nn/src/lib.rs`.

- Input must be 2D `[batch, dim]` for normalization; other ranks pass through unchanged.
- Default init: γ = 1, β = 0 (see `LayerNorm::new`).

Unit tests:

| Behavior | Test |
|----------|------|
| Row mean ≈ 0 after norm+affine at γ=1, β=0 | `layernorm_row_mean_near_zero` |
| GELU backward finite-diff parity | `gelu_backward_matches_finite_diff` |
| LN backward finite-diff parity (one-hot `grad_out`) | `layernorm_backward_matches_finite_diff` |

## Backward

`layernorm_backward` in `mmn-nn` computes `grad_x`, `grad_gamma`, `grad_beta` for `[batch, dim]` inputs. Used from `TransformerBlock::backward_attn_ffn` for both `ln1` and `ln2`.

## Training (`train_step_lm`)

| Parameter | Updated in LM step | Test |
|-----------|-------------------|------|
| `ln1.gamma` / `ln1.beta` | **yes** (`Train`) | `train_step_updates_layernorm_gamma`, `test_train_changes_ln1_gamma` |
| `ln2.gamma` / `ln2.beta` | **yes** (`Train`) | same |

Under `RL`, γ/β stay fixed (`test_rl_does_not_change_ln1_gamma`). `SPIN` runs `Train` and updates LN via the Train phase (`test_spin_changes_ln1_gamma`).

## IO / merge / quantize

All four tensors per block are checkpoint keys: `blocks.{i}.ln1.{gamma,beta}`, `blocks.{i}.ln2.{gamma,beta}`.

- Merge averages γ/β element-wise (same as linear weights).
- Quantize runs on γ/β; at default init bytes may be unchanged — see [quantize_coverage.md](quantize_coverage.md).

## Roadmap

- ~~Optional: finite-diff tests for `grad_gamma` / `grad_beta`~~ done pass 83
- Optional: fuse LN with residual in autograd tape

See [attention_coverage.md](attention_coverage.md) and [limitations.md](limitations.md).
