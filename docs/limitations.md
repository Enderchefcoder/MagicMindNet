# Known limitations (alpha)

MagicMindNet is a from-scratch training stack. The following gaps are intentional for the current alpha; see [CHANGELOG.md](../CHANGELOG.md) for fixes over time.

## Training

- Tokenization defaults to byte-level (`bytes % vocab_size`). Opt-in **BPE** or **unigram** (`UnigramEncoder`): train on QA/corpus text and pass `bpe_encoder=` or `unigram_encoder=` to `Train()` (`crates/mmn-data/src/bpe.rs`, `unigram.rs`, `tests/test_bpe_tokenizer_py.py`, `tests/test_unigram_tokenizer_py.py`).
- Training sequences use the model's **`max_seq_len`** (no longer hard-capped at 32 tokens).
- Token embeddings use **sinusoidal position encoding** by default (runtime, not checkpointed). Opt-in **learned `pos_embed`** or **RoPE** (`use_rope=True`) — see [position_encoding_coverage.md](position_encoding_coverage.md).
- Each `TransformerBlock` uses **two residuals**: `x2 = x + attn(ln1(x))` and `out = x2 + ffn(ln2(x2))` (pass 83). Backward routes skip grads through both adds.
- **Glint-style knobs** (defaults preserve classic Chatbot): `n_loops` (reuse the block stack with shared weights), `norm="rms"` (RMSNorm), `ffn="swiglu"` (gate×up SiLU), `tie_embeddings`, `loop_embed` (per-loop additive embedding), `final_norm` (optional final LayerNorm/RMSNorm), `lora_rank` (LoopLoRA QKV adapters; `0` = off, zero-init up = identity at init).
- Autoset presets: `sub-1M` / `sub-10M` / `sub-50M` / `sub-100M` / `sub-1B` / `sub-10B` for easy parameter sizing.
- CoT: `DatasetQA(cot=True)` with empty `thinktag` defaults to `<think>…</think>` wrappers;
  pass `thinktag="reason"` or `thinktag="a|b"` for custom open/close tags.- `TrainClassifier` updates backbone + head with CE; byte features are not a production text encoder.
- RL/SPIN use heuristic rewards, not environment rollouts.
- `TrainConfig.batch_size` on `Train()` and `TrainClassifier()` accumulates gradients over that many micro-batches (QA rows, corpus rows, or labeled classification rows) before one optimizer step (`batch_size=1` applies each step immediately).
- `Train()` accepts `DatasetQA` (aligned input→output) or `DatasetCorpus` (next-token LM on each row).
- RoPE rotates Q/K in attention (`use_rope=True`); mutually exclusive with learned `pos_embed` (see [position_encoding_coverage.md](position_encoding_coverage.md))
- `Chatbot(vision=True)` includes RGB conv patch prefix + text→image cross-attention after block 0 (see [vision_coverage.md](vision_coverage.md)); QA training loads optional `image` paths from disk.

## IO

- `mmn-safetensors-v1` (Chatbot) and `mmn-classifier-v1` (Classifier) are JSON checkpoints with tensor blobs.
- **`hf-safetensors`** / **`mmn-hf-safetensors-v1`**: Chatbot binary interchange (F32/F16/BF16→F32, **native GQA**). **`mmn-hf-classifier-v1`** for Classifier. `import_*("safetensors", …)` auto-detects binary vs JSON.
- `bin` / `mmn-bin-v1` stores architecture meta only (no weights), including optional `use_learned_pos_embed` / `max_seq_len`. Use `safetensors` or `hf-safetensors` for weight roundtrips.
- LayerNorm γ/β are included in `mmn-safetensors-v1` checkpoints per transformer block.

## CUDA

- Default builds use CPU parity in `mmn-cuda`. Real GPU GEMM requires `maturin develop --features cuda` and a CUDA toolkit.

## Diffusion

- Foundation VAE encode/decode + UNet noise prediction with `TrainDiffusion` on `DatasetImageGen` (8×8 RGB). `sample_rgb_patch` runs a simplified reverse-diffusion loop — not a production scheduler or full-resolution sampler.
- Checkpoints: `export_diffusion` / `import_diffusion` / `merge_diffusion` / `quantize_diffusion` (`mmn-diffusion-v1` JSON).
- Inpainting: `TrainDiffusion` on `DatasetImageEdit`, `sample_inpaint_rgb_patch`, masked denoise loss.

## Layer norm / attention

- Layer norm is implemented for 2D `[batch, dim]` tensors used in blocks. **γ/β are trained** in `Train()` / SPIN Train phase (pass 82) — see [layernorm_coverage.md](layernorm_coverage.md). Opt-in **RMSNorm** via `norm="rms"` (γ only; β unused).
- Multi-head attention forward is **scaled dot-product** with **causal masking** by default (`mmn-nn`, pass 84). **`Train()` updates attn q/k/v/out** (pass 81) and **LN γ/β** (pass 82).
- FFN defaults to GELU; opt-in **SwiGLU** via `ffn="swiglu"` (`blocks.N.ffn_gate` + `ffn_kind` meta).
- `n_loops>1` reuses shared block weights; KV-cache generation falls back to full forward when `n_loops>1`.
- `RL` updates `lm_head` only; `SPIN` alternates `Train` (FFN/embed/**attn**/LN) + `selfplay` RL. RL keeps attn/LN frozen — regression: `tests/test_train_rl_spin_py.py`.

### Roadmap (post-alpha)

| Gap | Coverage doc | Planned work |
|-----|--------------|--------------|
| Scaled dot-product attention backward | [attention_coverage.md](attention_coverage.md) | ~~done pass 81~~ |
| LayerNorm γ/β training | [layernorm_coverage.md](layernorm_coverage.md) | ~~done pass 82~~ |
| Glint LoopLoRA QKV + final_norm | this doc § Training | ~~done~~ (`lora_rank`, `final_norm`; GGUF `output_norm`) |
| Production tokenizer | this doc § Training | ~~BPE trainer~~; **UnigramEncoder** (`mmn-unigram-v1`) + Viterbi; SentencePiece-scale vocab next |
| Vision encoder | [vision_coverage.md](vision_coverage.md) | ~~DatasetQA image file loading~~; multi-patch tiles done |
| HF binary safetensors | this doc § IO | ~~Chatbot + Classifier export/import~~; ~~native GQA forward~~ done |
| Feature parity Wave 1 | [feature_parity.md](feature_parity.md) | ~~typical_p / Mirostat v2 / stream / embed / cosine LR / ChatML~~ |
| Feature parity Wave 2+ | [feature_parity.md](feature_parity.md) | grammars, tools, speculative decode (planned) |

See also [optimizers_coverage.md](optimizers_coverage.md) (Muon matrix routing) and [training_coverage.md](training_coverage.md).
