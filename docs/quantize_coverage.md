# Quantize coverage

`quantize(bot, "int8"|"int4")` and `quantize_classifier(clf, …)` round each float weight to a fixed grid (`scale=127` for int8, `15` for int4).

## Chatbot tensors quantized

| Tensor group | int8 weight change test | int4 weight change test |
|--------------|-------------------------|-------------------------|
| `embed`, `lm_head` | Rust + matrix | Rust + matrix |
| `pos_embed` (learned PE) | Rust + pytest | Rust + pytest |
| Block attn `q/k/v/out` | Rust + matrix | Rust + matrix |
| Block `ffn`, `ffn2` | Rust + matrix | Rust + matrix |
| Block `ln1.gamma`, `ln1.beta`, `ln2.gamma`, `ln2.beta` | Rust¹ + Python² | Rust¹ + Python² |

¹ **Non-default γ/β** — at init (γ=1, β=0) int8/int4 grids leave values unchanged; `quantize_int8_changes_block_ln1_gamma_when_non_default` and `quantize_int4_changes_block_ln2_beta_when_non_default` in `mmn-io`, plus `tests/test_io_ln_quantize_py.py`, prove mutation when γ≠1 or β≠0.

² Python tests tamper checkpoint JSON before import, then quantize and re-export.

## Classifier

| Key | int8 | int4 |
|-----|------|------|
| `backbone` | matrix | matrix |
| `head` | matrix | matrix |

See `tests/test_io_classifier_matrix_py.py` and `tests/test_quantize_classifier.py`.

## Meta preserved

| Behavior | Test |
|----------|------|
| Unknown mode error | `test_quantize_unknown_mode.py` |
| Getters unchanged (`vocab_size`, etc.) | `test_quantize_preserves_getters.py` |
| `has_vision` after quantize | Rust + `test_vision_chatbot_py.py` |
| Learned PE meta after quantize | — | `test_quantize_learned_pos_embed_meta_unchanged` |
| Learned PE mean loss drift &lt;50% | Rust | `test_learned_pos_embed_quantize_py.py` |
| Learned PE quantize after train | — | `test_quantize_int8/int4_learned_pos_embed_after_train_within_tolerance` |
| Learned PE quantize after train (Rust) | `train_learned_pos_embed_quantize_int8_preserves_mean_loss` | — |
| Classifier labels / `input_dim` | `test_quantize_classifier.py` |

## Matrix tests

- Chatbot: `tests/test_io_checkpoint_matrix_py.py` (8 linear keys × quantize export change; LN excluded from export-byte matrix by design)
- Classifier: `tests/test_io_classifier_matrix_py.py`

Full key list: [checkpoint_coverage.md](checkpoint_coverage.md).

## Running

```powershell
cargo test -p mmn-io quantize
pytest tests/test_io_ln_quantize_py.py tests/test_io_checkpoint_matrix_py.py -q
```
