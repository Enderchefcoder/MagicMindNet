# Checkpoint formats

MagicMindNet uses two Chatbot weight formats:

1. **`mmn-safetensors-v1`** — JSON wrapper with little-endian F32 tensor blobs (default `export(..., "safetensors", …)`).
2. **`mmn-hf-safetensors-v1`** — Hugging Face **binary** safetensors (`export(..., "hf-safetensors", …)`). Same MMN tensor keys and meta; readable by HF tooling.

`import_model("safetensors", [path])` auto-detects binary vs JSON from the first byte.

Export creates missing parent directories for the output path (e.g. `checkpoints/run1/model.mmn`).

## Chatbot — `mmn-safetensors-v1` (JSON)

```python
ai.export(bot, "safetensors", "model.mmn")
bot2 = ai.import_model("safetensors", ["model.mmn"])
```

## Chatbot — `mmn-hf-safetensors-v1` (binary)

```python
ai.export(bot, "hf-safetensors", "model.safetensors")
bot2 = ai.import_model("hf-safetensors", ["model.safetensors"])
# or: ai.import_model("safetensors", ["model.safetensors"])  # auto-detect
```

Header `__metadata__` includes `format: mmn-hf-safetensors-v1` and a JSON `meta` string (`vocab_size`, `n_layer`, `d_model`, optional `ffn_dim`, vision flags, etc.). Import also accepts common HF tensor name aliases (`model.layers.N.self_attn.q_proj.weight`, `transformer.wte.weight`, …) and adapts external layouts:

- **GPT-2 fused QKV** (`attn.c_attn`) → split/transpose into `blocks.N.attn.{q,k,v}`
- **Llama SwiGLU** (`mlp.gate_proj` + `mlp.up_proj`) → element-wise product into `blocks.N.ffn`; `mlp.down_proj` → `ffn2`
- **Tied embeddings** — missing `lm_head` copies `embed`
- **RMSNorm-only** checkpoints — missing LayerNorm β defaults to 0, γ to 1
- **GQA** — smaller `k_proj`/`v_proj` expanded to MMN MHA shape using `num_attention_heads` / `num_key_value_heads` in meta (or head-dim heuristic)
- **F16 / BF16** weights decoded to F32 on import

### Chatbot — `mmn-bin-v1` (architecture stub only)

```python
ai.export(bot, "bin", "arch.bin")  # meta: vocab_size, n_layer, d_model, vision, optional PE flags
bot2 = ai.import_model("bin", ["arch.bin"])  # fresh weights; shape + PE getters match export
assert (bot2.vocab_size, bot2.n_layer, bot2.d_model) == (bot.vocab_size, bot.n_layer, bot.d_model)
```

`bin` also stores `use_learned_pos_embed` and `max_seq_len` when learned PE is enabled (no weights).

Use `safetensors` for full weight roundtrips. `bin` rejects `mmn-safetensors-v1` / classifier files. An empty `{}` bin stub loads defaults: `vocab_size=32000`, `d_model=128`, `n_layer=4`, `vision=false`, sinusoidal PE.

**Meta:** `vocab_size`, `n_layer`, `d_model`, `vision`, optional `seed` (init RNG; weights are authoritative after import), optional `use_learned_pos_embed` / `max_seq_len`. Import **requires** `vocab_size`, `n_layer`, and `d_model`; tensor shapes must match meta (`embed` `[vocab_size, d_model]`, `lm_head` `[vocab_size, d_model]`, block attn `[d_model, d_model]`, `ffn` `[ffn_dim, d_model]`, `ffn2` `[d_model, ffn_dim]`, layernorm `[d_model]` where `ffn_dim = d_model * 4`). If `n_layer` in meta exceeds exported block tensors, import fails on the first missing `blocks.{i}.*` key.

**Tensors:** `embed`, `lm_head`, optional `pos_embed` `[max_seq_len, d_model]` when learned PE is on, per-block `blocks.{i}.attn.{q,k,v,out}`, `ffn`, `ffn2`, `ln1.{gamma,beta}`, `ln2.{gamma,beta}`

## Classifier — `mmn-classifier-v1`

```python
ai.export_classifier(clf, "safetensors", "clf.mmn")
clf2 = ai.import_classifier("safetensors", ["clf.mmn"])
assert clf2.input_dim == clf.input_dim and clf2.num_labels == clf.num_labels
```

**Meta:** `input_dim`, `labels` (non-empty string list), optional `seed` — all required fields are validated; tensor shapes must match (`backbone` `[128, input_dim]`, `head` `[num_labels, 128]`).

**Tensors:** `backbone`, `head`

## Quantization

```python
ai.quantize(bot, "int8")      # Chatbot — also `"int4"`
ai.quantize_classifier(clf, "int8")  # also `"int4"`
```

## Merge

```python
merged_bot = ai.merge(bot_a, bot_b)  # same n_layer, d_model, vocab; element-wise mean of embed, lm_head, all block tensors (attn, ffn, layernorm); keeps first init_seed
merged_clf = ai.merge_classifier(clf_a, clf_b)  # same labels + input_dim; averages backbone + head; keeps first init_seed
```

Raises `ModelMismatchError` when shapes or label sets differ.

## BytePairEncoder — `mmn-bpe-v1`

```python
bpe = ai.BytePairEncoder.train(["hello hello"], vocab_size=512, num_merges=32)
bpe.save("tokenizer.mmn")
bpe2 = ai.BytePairEncoder.load("tokenizer.mmn")
assert bpe2.encode("hello") == bpe.encode("hello")
```

**Meta:** `format` (`mmn-bpe-v1`), `vocab_size` (≥ 256), `merges` (list of `[left, right]` byte or merged token ids). BPE checkpoints are separate from Chatbot weights — pass the loaded encoder to `Train()` / `compute_mean_loss()`.

Chatbot `mmn-safetensors-v1` may reference a sibling sidecar via `meta.bpe_checkpoint` (e.g. `bot.bpe.mmn`). Use `export(bot, "safetensors", path, bpe_encoder=enc)` to write both files, then `load_bpe_sidecar(path)` after import.

See [checkpoint_coverage.md](checkpoint_coverage.md) for the full chatbot tensor regression matrix (100% key coverage).

Rust `mmn-io` modules: `chatbot_io`, `classifier_io`, `block_tensors`, `checkpoint_util`, `tensor_merge`. Regression tests live in `io_tests/` (`chatbot_io_tests`, `classifier_io_tests`).

## Format guards

`import_model` accepts only `format: "mmn-safetensors-v1"`. `import_classifier` accepts only `mmn-classifier-v1`. Cross-import raises a clear error (see `tests/test_import_format_guard.py`).
