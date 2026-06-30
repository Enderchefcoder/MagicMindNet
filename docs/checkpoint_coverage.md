# Checkpoint IO coverage matrix

This document tracks **100% regression coverage** of the chatbot `mmn-safetensors-v1` strict IO contract: every exported tensor key is tested for missing-key rejection, shape mismatch, merge averaging, and quantize mutation.

Classifier coverage is documented separately (backbone/head + meta); see `tests/test_import_classifier_strict_py.py` and related files.

## Chatbot tensor keys (single block)

| Tensor key | Missing import | Shape mismatch | Merge average | int8 quantize | int4 quantize |
|------------|----------------|----------------|---------------|---------------|---------------|
| `embed` | Rust + matrix | Rust + matrix | Rust + matrix | Rust + matrix | Rust + matrix |
| `lm_head` | Rust + matrix | Rust + matrix | Rust + matrix | Rust + matrix | Rust + matrix |
| `pos_embed` (learned PE only) | Rust + pytest | Rust + pytest | Rust + pytest | Rust + matrix + loss¹ | Rust + matrix + loss¹ |
| `blocks.0.attn.q` | Rust + matrix | Rust + matrix | Rust + matrix | Rust + matrix | Rust + matrix |
| `blocks.0.attn.k` | Rust + matrix | Rust + matrix | Rust + matrix | Rust + matrix | Rust + matrix |
| `blocks.0.attn.v` | Rust + matrix | Rust + matrix | Rust + matrix | Rust + matrix | Rust + matrix |
| `blocks.0.attn.out` | Rust + matrix | Rust + matrix | Rust + matrix | Rust + matrix | Rust + matrix |
| `blocks.0.ffn` | Rust + matrix | Rust + matrix | Rust + matrix | Rust + matrix | Rust + matrix |
| `blocks.0.ffn2` | Rust + matrix | Rust + matrix | Rust + matrix | Rust + matrix | Rust + matrix |
| `blocks.0.ln1.gamma` | Rust + matrix | Rust + matrix | Rust + matrix | code path¹ | code path¹ |
| `blocks.0.ln1.beta` | Rust + matrix | Rust + matrix | Rust + matrix | code path¹ | code path¹ |
| `blocks.0.ln2.gamma` | Rust + matrix | Rust + matrix | Rust + matrix | code path¹ | code path¹ |
| `blocks.0.ln2.beta` | Rust + matrix | Rust + matrix | Rust + matrix | code path¹ | code path¹ |

¹ `quantize_model` runs on layernorm tensors; at default init (γ=1, β=0) exported bytes are unchanged — see `quantize_int8_changes_block_ln1_gamma_when_non_default` (Rust) and `tests/test_io_ln_quantize_py.py` for non-default γ/β mutation.

² Learned `pos_embed` is optional (`use_learned_pos_embed` meta). Quantize mutates weights (`quantize_int8/int4_changes_pos_embed_weights`); forward/mean loss stays finite with &lt;50% relative drift (`quantize_*_learned_pos_embed_preserves_forward_loss_within_tolerance`, `test_learned_pos_embed_quantize_py.py`).

Full quantize matrix: [quantize_coverage.md](quantize_coverage.md).

**Matrix test file:** `tests/test_io_checkpoint_matrix_py.py` (12 keys × 4 behaviors + 24 quantize cases; learned `pos_embed` adds 4 behaviors + 2 quantize modes).

### Python test helpers (`tests/conftest.py`)

Matrix and merge tests share checkpoint introspection helpers (pass 68–70):

| Helper | Use |
|--------|-----|
| `load_checkpoint(path)` | Full JSON (`meta` + `tensors`) for tamper tests |
| `load_checkpoint_tensors(path)` | Tensor map only — merge/quantize byte compares |
| `tensor_entry_first_f32(entry)` | First f32 in a tensor dict — merge average checks |
| `tamper_tensor_entry_first_f32(entry, value)` | Non-default γ/β for LN quantize tests |
| `checkpoint_tensor_bytes(path, key)` | Byte equality — frozen train/RL regressions |

Regression: `tests/test_conftest_helpers_py.py`.

## Learned `pos_embed` (optional)

Present when `use_learned_pos_embed=true` in meta. Shape `[max_seq_len, d_model]`.

| Behavior | Rust | Python |
|----------|------|--------|
| Roundtrip weights | `learned_pos_embed_roundtrip_preserves_weights` | `test_learned_pos_embed_io_py.py` |
| Missing tensor strict | `import_rejects_missing_pos_embed_when_meta_requires` | — |
| Shape vs `max_seq_len` | `import_rejects_pos_embed_shape_mismatch` | — |
| Merge average | `merge_models_averages_pos_embed` | `test_merge_learned_pos_embed_averages_weights`, `test_merge_trained_learned_pos_embed_averages_weights` |
| Merge PE settings mismatch | `merge_rejects_pos_embed_settings_mismatch` | `test_merge_rejects_learned_vs_sinusoidal_pos_embed` |
| int8/int4 weight mutation | `quantize_int8/int4_changes_pos_embed_weights` | `test_io_checkpoint_matrix_py.py` (learned PE) |
| Quantize loss tolerance | `quantize_int8/int4_learned_pos_embed_preserves_forward_loss_within_tolerance` | `test_learned_pos_embed_quantize_py.py` |
| Quantize meta preserved | — | `test_quantize_learned_pos_embed_meta_unchanged` |
| `bin` architecture meta | `bin_learned_pos_embed_roundtrip_preserves_meta` | `test_bin_shape_getters.py` |

See [position_encoding_coverage.md](position_encoding_coverage.md).

## Meta and guards (chatbot)

| Behavior | Rust | Python |
|----------|------|--------|
| Missing `vocab_size` / `n_layer` / `d_model` | yes | yes |
| `n_layer` vs block count mismatch | yes | yes |
| Invalid / empty JSON | yes | yes |
| Tensor data length mismatch | yes | yes |
| Non-numeric tensor bytes | yes | — |
| Wrong format / classifier cross-import | yes | yes |
| First path only in file list | — | yes |
| Roundtrip preserves weights / loss | yes | yes |
| Learned PE loss after import | `import_preserves_forward_loss_learned_pos_embed` | `test_export_import_preserves_learned_pos_embed_compute_loss` |
| Learned PE loss after train + export | — | `test_export_import_preserves_learned_pos_embed_after_train` |
| Learned PE train→export→import (Rust) | `train_learned_pos_embed_export_import_preserves_mean_loss`, `train_corpus_learned_pos_embed_export_import_preserves_mean_loss` | — |
| RoPE loss after import | `import_preserves_forward_loss_rope` | `test_rope_export_import_preserves_mean_loss` |
| RoPE train→export→import (Rust) | `train_rope_export_import_preserves_mean_loss` | `test_rope_export_import_preserves_mean_loss` |
| `bin` RoPE meta | `bin_rope_roundtrip_preserves_meta` | — |

## Classifier (`mmn-classifier-v1`)

| Tensor key | Missing import | Shape mismatch | Merge average | int8 quantize | int4 quantize |
|------------|----------------|----------------|---------------|---------------|---------------|
| `backbone` | Rust + matrix | Rust + matrix | Rust + matrix | Rust + matrix | Rust + matrix |
| `head` | Rust + matrix | Rust + matrix | Rust + matrix | Rust + matrix | Rust + matrix |

**Matrix test file:** `tests/test_io_classifier_matrix_py.py` (2 keys × 4 behaviors + 4 quantize cases).

| Behavior | Rust | Python |
|----------|------|--------|
| Missing labels / input_dim | yes | yes |
| Label mismatch on merge | yes | yes |
| Invalid / empty JSON | yes | yes |
| Chatbot cross-import | yes | yes |

## Multi-block chatbot (`n_layer > 1`)

| Behavior | Rust | Python |
|----------|------|--------|
| `n_layer` meta vs block count mismatch | yes | yes |
| Missing `blocks.1.*` tensors (all 10 keys) | partial + matrix | matrix |
| Merge averages `blocks.1` weights | yes | yes |
| Two-layer roundtrip preserves `n_layer` | — | yes |

**Multi-block test file:** `tests/test_io_multiblock_chatbot_py.py`

## Diffusion (`mmn-diffusion-v1`)

Seven conv weight tensors: `vae_enc_conv1`, `vae_enc_conv2`, `vae_dec_conv1`, `vae_dec_conv2`, `unet_down`, `unet_mid`, `unet_up`. Meta: `latent_channels`, `spatial`.

| Tensor key | Missing import | Shape mismatch | Merge average | int8 quantize | int4 quantize |
|------------|----------------|----------------|---------------|---------------|---------------|
| All 7 keys | `test_io_diffusion_matrix_py.py` | matrix (same element count) | matrix | matrix (subset) | matrix (subset) |

| Behavior | Rust | Python |
|----------|------|--------|
| Roundtrip preserves sample | `diffusion_export_import_roundtrip_preserves_sample` | `test_diffusion_export_import_roundtrip` |
| Reject chatbot checkpoint | `import_diffusion_rejects_chatbot_checkpoint` | `test_import_diffusion_rejects_chatbot_checkpoint` |
| Quantize export/import sample | — | `test_quantize_diffusion_export_import_preserves_sample` |
| Merge `latent_channels` mismatch | `merge_diffusion_rejects_latent_channel_mismatch` | — |

**Matrix test file:** `tests/test_io_diffusion_matrix_py.py`

See [diffusion_coverage.md](diffusion_coverage.md).

## Running coverage checks

```powershell
.\scripts\verify_gate.ps1
pytest tests/test_io_checkpoint_matrix_py.py -q
cargo test -p mmn-io
```

After changes, update counts in `README.md` via `.\scripts\count_tests.ps1`.
