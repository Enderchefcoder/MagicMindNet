# Position encoding coverage

MagicMindNet supports three position-encoding modes on `Chatbot`:

| Mode | Flag | Checkpoint | Trained |
|------|------|------------|---------|
| Sinusoidal (default) | `use_learned_pos_embed=False`, `use_rope=False` | none (runtime) | no |
| Learned table | `use_learned_pos_embed=True` | `pos_embed` `[max_seq_len, d_model]` | yes |
| Rotary (RoPE) | `use_rope=True` | meta `rope_theta` only | no extra weights |

`use_learned_pos_embed` and `use_rope` are mutually exclusive.

## Sinusoidal (default)

- `sinusoidal_position_encoding` / `add_sinusoidal_position_encoding` in `crates/mmn-nn/src/lib.rs`
- Applied after `embed.forward` in `forward_hidden` / `backward_lm_grads`
- Gradients do not flow into PE (fixed)

## Learned `pos_embed` (opt-in)

- `Chatbot::new_with_pe_options(..., use_learned_pos_embed, max_seq_len)` — default `max_seq_len=512`
- Python: `Chatbot(..., use_learned_pos_embed=True, max_seq_len=64)`
- IO meta: `use_learned_pos_embed`, `max_seq_len`; tensor key `pos_embed`
- `train_step_lm` updates `pos_embed` via `embedding_backward` on position indices `0..seq-1`
- merge / quantize include `pos_embed` when enabled

## RoPE (opt-in)

- `apply_rope` / `apply_rope_backward` in `mmn-nn`; rotates Q/K per head after projection
- `Chatbot::new_with_position_options(..., use_rope, rope_theta)` — default `rope_theta=10000`
- Python: `Chatbot(..., use_rope=True, rope_theta=10000.0)`
- Skips additive sinusoidal/learned PE on embeddings
- IO meta: `use_rope`, `rope_theta`; no extra checkpoint tensors
- merge requires matching `use_rope` and `rope_theta`

## Tests

| Behavior | Test |
|----------|------|
| PE differs by row index | `position_encoding_tests::sinusoidal_pe_differs_by_position` |
| RoPE changes Q/K | `position_encoding_tests::rope_changes_qk_values` |
| RoPE backward | `position_encoding_tests::rope_backward_matches_finite_diff` |
| RoPE skips additive PE | `chatbot_tests::rope_skips_additive_position_encoding` |
| RoPE changes loss | `chatbot_tests::rope_attention_differs_from_no_rope` |
| RoPE IO meta | `safetensors_rope_meta_roundtrip`, `test_rope_chatbot_py.py` |
| RoPE trains | `test_rope_trains_and_reduces_loss` |
| RoPE corpus train | `train_corpus_rope_reduces_mean_loss` |
| RoPE merge mismatch | `test_merge_rejects_rope_theta_mismatch`, `test_merge_rejects_rope_vs_sinusoidal` |
| `benchmark_train --rope` | `test_benchmark_train_rope_example_runs` |
| `eval_mean_loss` `--rope` | `test_eval_mean_loss_qa_rope_runs`, `test_eval_mean_loss_corpus_rope_runs` |
| `eval_mean_loss --rope --train` | `test_eval_mean_loss_qa_rope_train_runs`, `test_eval_mean_loss_corpus_rope_train_runs` |
| `corpus_benchmark --rope` | `test_corpus_benchmark_rope_example_runs` |
| `quickstart --rope` | `test_quickstart_rope_example_runs` |
| Chatbot forward uses PE | `chatbot_tests::position_encoding_affects_forward_hidden` |
| Learned PE trains | `chatbot_tests::train_step_updates_learned_pos_embed` |
| IO roundtrip | `learned_pos_embed_roundtrip_preserves_weights`, `test_learned_pos_embed_io_py.py` |
| Import loss parity | `import_preserves_forward_loss_learned_pos_embed`, `test_export_import_preserves_learned_pos_embed_compute_loss` |
| Example smoke | `examples/learned_pos_embed_roundtrip.py`, `test_learned_pos_embed_roundtrip_example_runs` |
| Example smoke (train first) | `learned_pos_embed_roundtrip.py --train`, `test_learned_pos_embed_roundtrip_train_example_runs` |
| Merge averages `pos_embed` | `merge_models_averages_pos_embed`, `test_merge_learned_pos_embed_averages_weights`, `test_merge_trained_learned_pos_embed_averages_weights` |
| Merge PE mismatch | `merge_rejects_pos_embed_settings_mismatch`, `test_merge_rejects_learned_vs_sinusoidal_pos_embed` |
| Quantize `pos_embed` | `quantize_int8_changes_pos_embed_weights`, `quantize_int4_changes_pos_embed_weights` |
| Train updates learned PE | `test_train_changes_learned_pos_embed` |
| RL frozen on learned PE | `test_rl_does_not_change_learned_pos_embed` |
| SPIN updates learned PE | `test_spin_changes_learned_pos_embed` |
| `parameters()` includes PE | `learned_pos_embed_increases_parameter_count`, `test_learned_pos_embed_increases_parameters` |
| Import `pos_embed` shape | `import_rejects_pos_embed_shape_mismatch` |
| `bin` PE meta roundtrip | `bin_learned_pos_embed_roundtrip_preserves_meta`, `test_bin_roundtrip_preserves_learned_pos_embed_meta` |
| `max_seq_len` guard | `learned_pos_embed_rejects_long_sequence`, `test_compute_loss_rejects_sequence_longer_than_max_seq_len` |
| Corpus `Train()` + learned PE | `train_corpus_updates_learned_pos_embed`, `test_train_corpus_learned_pos_embed_reduces_mean_loss` |
| Corpus benchmark learned PE | `corpus_benchmark.py --learned-pe`, `test_corpus_benchmark_learned_pe_example_runs` |
| eval_mean_loss learned PE | `eval_mean_loss.py qa|corpus --learned-pe`, `test_eval_mean_loss_*_learned_pe_runs` |
| eval_mean_loss train + learned PE | `eval_mean_loss.py --train --learned-pe`, `test_eval_mean_loss_*_learned_pe_train_runs` |
| benchmark_train learned PE | `benchmark_train.py --learned-pe`, `test_benchmark_train_learned_pe_example_runs` |
| Export after train (learned PE) | — | `test_export_import_preserves_learned_pos_embed_after_train` |
| Train → export → import (Rust) | `train_learned_pos_embed_export_import_preserves_mean_loss` | — |
| Post-import corpus/QA train | `test_train_corpus_after_import_learned_pos_embed_reduces_loss`, `test_train_after_import_learned_pos_embed_reduces_loss` |
| Quantize loss tolerance | `quantize_int8/int4_learned_pos_embed_preserves_forward_loss_within_tolerance`, `test_learned_pos_embed_quantize_py.py` |
| Quantize after train | — | `test_quantize_int8_learned_pos_embed_after_train_within_tolerance` |
| Quantize meta preserved | `test_quantize_learned_pos_embed_meta_unchanged` |
| Mean QA loss decreases | `test_learned_pos_embed_compute_mean_loss_decreases_after_train` |
| IO matrix (learned PE) | `test_import_rejects_missing_learned_pos_embed_matrix_py`, merge/quantize matrix |
| Missing `pos_embed` strict | `import_rejects_missing_pos_embed_when_meta_requires` |
| Python getters | `test_mmn_py_bindings_py.py` |

## Roadmap

- Learned RoPE frequency scale (optional trainable θ per layer)
- Long-context scaling (NTK / YaRN-style) for `max_seq_len` beyond training

See [attention_coverage.md](attention_coverage.md) and [limitations.md](limitations.md).
