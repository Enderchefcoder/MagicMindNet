# Optimizers & autograd coverage

Regression coverage for `mmn-optim` (AdamW, Muon, Hybrid, GradAccumulator) and `mmn-core` autograd/ops used in training.

## Optimizers (`mmn-optim`)

| Behavior | Test |
|----------|------|
| AdamW updates weights | `adamw_changes_weights` |
| Muon Newton-Schulz finite output | `newton_schulz_output_finite` |
| Hybrid routes 2D params to Muon | `classify_param_matrix_vs_other`, `hybrid_routes_matrix_to_muon`, `muon_step_matrix_changes_weights` |
| Hybrid routes vectors to AdamW | `hybrid_vector_step_uses_adamw` |
| GradAccumulator averages micro-batches | `grad_accumulator_averages_two_micro_batches` |
| GradAccumulator `clear()` resets | `grad_accumulator_clear_resets_state` |

**Training integration:** `batch_size>1` accumulation is exercised in `mmn-train` (`train_batch_size_two_accumulates_and_reduces_loss`) and Python `test_train_batch_size.py`.

Default `TrainConfig.optimizer` is `"hybrid"` (Muon on 2D weight matrices, AdamW on biases/embed rows).

## Autograd & ops (`mmn-core`)

| Behavior | Crate | Test |
|----------|-------|------|
| Tape records backward | `mmn-core` | `tape_accumulates_grad` |
| `add` backward splits grad to parents | `mmn-core` | `backward_add_splits_grad_to_parents` |
| CE grad at target class negative | `mmn-core` | `ce_grad_pushes_down_target_class` |
| CE grad batch mean (rows sum ~0) | `mmn-core` | `ce_grad_averages_over_batch` |
| `linear_backward` shape contract | `mmn-core` | `linear_backward_grad_shapes_match` |
| Embedding grad accumulates rows | `mmn-core` | `embedding_backward_accumulates_rows` |
| GELU backward finite diff | `mmn-nn` | `gelu_backward_matches_finite_diff` |
| LayerNorm row mean ~0 | `mmn-nn` | `layernorm_row_mean_near_zero` |

## Tensor primitives (`mmn-core`)

| Behavior | Test |
|----------|------|
| `matmul` shapes | `matmul_shapes` |
| `softmax` rows sum to 1 | `softmax_rows_sum_to_one` |
| `relu` zeros negatives | `relu_zeros_negatives` |

## Running

```powershell
cargo test -p mmn-optim
cargo test -p mmn-core
cargo test -p mmn-nn
.\scripts\verify_gate.ps1
```

See [training.md](training.md) and [training_coverage.md](training_coverage.md).
