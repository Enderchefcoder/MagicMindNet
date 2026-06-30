# Known limitations (alpha)

MagicMindNet is a from-scratch training stack. The following gaps are intentional for the current alpha; see [CHANGELOG.md](../CHANGELOG.md) for fixes over time.

## Training

- Tokenization defaults to byte-level (`bytes % vocab_size`). Opt-in **BPE**: train `ai.BytePairEncoder` on QA/corpus text and pass `bpe_encoder=` to `Train()` (`crates/mmn-data/src/bpe.rs`, `tests/test_bpe_tokenizer_py.py`).
- Token embeddings use **sinusoidal position encoding** by default (runtime, not checkpointed). Opt-in **learned `pos_embed`** is checkpointed in `mmn-safetensors-v1` and architecture flags are in `mmn-bin-v1` — see [position_encoding_coverage.md](position_encoding_coverage.md).
- Each `TransformerBlock` uses **two residuals**: `x2 = x + attn(ln1(x))` and `out = x2 + ffn(ln2(x2))` (pass 83). Backward routes skip grads through both adds.
- `TrainClassifier` updates backbone + head with CE; byte features are not a production text encoder.
- RL/SPIN use heuristic rewards, not environment rollouts.
- `TrainConfig.batch_size` on `Train()` and `TrainClassifier()` accumulates gradients over that many micro-batches (QA rows, corpus rows, or labeled classification rows) before one optimizer step (`batch_size=1` applies each step immediately).
- `Train()` accepts `DatasetQA` (aligned input→output) or `DatasetCorpus` (next-token LM on each row).
- `Chatbot(vision=True)` is checkpoint metadata only — no image encoder forward path yet (see [vision_coverage.md](vision_coverage.md)).

## IO

- `mmn-safetensors-v1` (Chatbot) and `mmn-classifier-v1` (Classifier) are JSON checkpoints with tensor blobs. They are not Hugging Face binary safetensors.
- `bin` / `mmn-bin-v1` stores architecture meta only (no weights), including optional `use_learned_pos_embed` / `max_seq_len`. Use `safetensors` for weight roundtrips.
- LayerNorm γ/β are included in `mmn-safetensors-v1` checkpoints per transformer block.

## CUDA

- Default builds use CPU parity in `mmn-cuda`. Real GPU GEMM requires `maturin develop --features cuda` and a CUDA toolkit.

## Diffusion

- VAE/UNet exist structurally; conv paths may be identity/stub. Not a full Stable-Diffusion-class training pipeline.

## Layer norm / attention

- Layer norm is implemented for 2D `[batch, dim]` tensors used in blocks. **γ/β are trained** in `Train()` / SPIN Train phase (pass 82) — see [layernorm_coverage.md](layernorm_coverage.md).
- Multi-head attention forward is **scaled dot-product** with **causal masking** by default (`mmn-nn`, pass 84). **`Train()` updates attn q/k/v/out** (pass 81) and **LN γ/β** (pass 82).
- `RL` updates `lm_head` only; `SPIN` alternates `Train` (FFN/embed/**attn**/LN) + `selfplay` RL. RL keeps attn/LN frozen — regression: `tests/test_train_rl_spin_py.py`.

### Roadmap (post-alpha)

| Gap | Coverage doc | Planned work |
|-----|--------------|--------------|
| Scaled dot-product attention backward | [attention_coverage.md](attention_coverage.md) | ~~done pass 81~~ |
| LayerNorm γ/β training | [layernorm_coverage.md](layernorm_coverage.md) | ~~done pass 82~~ |
| Production tokenizer | this doc § Training | ~~BPE trainer + Python `BytePairEncoder` + `Train(bpe_encoder=)`~~; SentencePiece-scale vocab next |
| Vision encoder | [vision_coverage.md](vision_coverage.md) | Image → patch embed forward path |
| HF binary safetensors | this doc § IO | Optional interchange format |

See also [optimizers_coverage.md](optimizers_coverage.md) (Muon matrix routing) and [training_coverage.md](training_coverage.md).
