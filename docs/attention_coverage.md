# Attention path (alpha)

MagicMindNet blocks use **scaled dot-product self-attention** in `mmn-nn`. **`Train()` / `train_step_lm` backprop through attn and LayerNorm** (passes 81–82).

## Forward path (pass 80)

```
Q,K,V = linear(x)   each [seq_len, d_model]
per head: scores = QKᵀ / √head_dim  → causal mask (default) → softmax → weights @ V
merge heads → out_proj
```

Implementation: `crates/mmn-nn/src/lib.rs` (`scaled_dot_product_attention`, `MultiHeadAttention`, `TransformerBlock`).

## Block stack

```
embed → [block × n_layer] → lm_head
         └─ ln1 → scaled dot-product attn → residual
              → ln2 → FFN (ffn + gelu + ffn2) → residual
```

## What trains today (`train_step_lm`)

| Parameter group | Updated | Test |
|-----------------|---------|------|
| `embed` rows in batch | yes | `train_step_updates_embed_and_ffn2` |
| `lm_head` | yes | `mmn-train` loss tests |
| Block `ffn` / `ffn2` (all layers) | yes | `train_step_updates_all_blocks_ffn2` |
| Block `attn.{q,k,v,out}` | **yes** (`Train`) | `train_step_updates_attn_q_weights`, `test_train_block_params_py.py` (`test_train_changes_attn_q_weights`) |
| Block `ln1` / `ln2` γ/β | **yes** (`Train`) | `train_step_updates_layernorm_gamma`, `test_train_block_params_py.py` (`test_train_changes_ln1_gamma`) |

## RL / SPIN (lm_head + Train path)

`RL` updates **only** `lm_head`. `SPIN` alternates `Train` (FFN/embed/attn/LN) + `selfplay` RL (`lm_head`). Attention and LN stay fixed under RL:

| Mode | Attn updates | LN updates | Test |
|------|--------------|------------|------|
| `Train` | yes | yes | `test_train_changes_attn_q_weights`, `test_train_changes_ln1_gamma` |
| `RL` policy | no (lm_head only) | no | `test_rl_does_not_change_attn_q_weights`, `test_rl_does_not_change_ln1_gamma` |
| `SPIN` | yes (via `Train` phase) | yes (via `Train` phase) | `test_spin_changes_attn_q_weights`, `test_spin_changes_ln1_gamma` |

Gradients flow: CE → `lm_head` → hidden → **FFN backward** + **attn backward** + **LN backward**. RL skips block weights; SPIN runs full `Train` between RL steps.

## IO / merge / quantize

Attention tensors are full citizens in checkpoints (`blocks.{i}.attn.{q,k,v,out}`). See [checkpoint_coverage.md](checkpoint_coverage.md) and [quantize_coverage.md](quantize_coverage.md).

## Roadmap (not alpha)

### Milestones

- ~~Scaled dot-product attention forward per head~~ **done pass 80**
- ~~Wire attn grads into `train_step_lm`~~ **done pass 81**
- ~~LayerNorm γ/β backward~~ **done pass 82**
- ~~Causal self-attention mask (LM default)~~ **done pass 84**
- Optional: cross-attn for vision-flag models — see [vision_coverage.md](vision_coverage.md)

See [limitations.md](limitations.md) and [optimizers_coverage.md](optimizers_coverage.md).
