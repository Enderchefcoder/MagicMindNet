# MagicMindNet API Reference

Import as:

```python
import magicmindnet as ai
```

New here? Read [getting_started.md](getting_started.md) first.

**Related docs:** [training.md](training.md) · [training_coverage.md](training_coverage.md) · [checkpoints.md](checkpoints.md) · [checkpoint_coverage.md](checkpoint_coverage.md) · [dataset_coverage.md](dataset_coverage.md) · [examples/README.md](../examples/README.md)

---

## Table of contents

1. [Public exports (`__all__`)](#public-exports-__all__)
2. [Datasets](#datasets)
3. [Models](#models)
4. [Training](#training)
5. [Checkpoints & merge](#checkpoints--merge)
6. [Utilities](#utilities)
7. [Errors](#errors)
8. [Examples](#examples)

---

## Public exports (`__all__`)

Every name below is defined on `import magicmindnet as ai` and listed in `ai.__all__`.

| Category | Names |
|----------|--------|
| Version | `__version__` |
| Datasets | `DatasetQA`, `DatasetCorpus`, `DatasetClassification`, `DatasetImageGen`, `DatasetImageEdit` |
| Models | `Chatbot`, `Classifier`, `Diffusion` |
| Training | `TrainConfig`, `Train`, `TrainClassifier`, `TrainDiffusion`, `RL`, `SPIN` |
| IO | **`load`** (universal), `export`, `import_model`, `merge`, `quantize`, `export_classifier`, `import_classifier`, `merge_classifier`, `quantize_classifier`, `export_diffusion`, `import_diffusion`, `merge_diffusion` |
| Array IO | `save_npy`, `load_npy`, `save_npz`, `load_npz`, `save_pt`, `load_pt`, `load_h5`, `load_keras` (NumPy/PyTorch/TF interchange, no numpy/torch/h5py needed) |
| GGUF tools | `gguf_info`, `load_gguf_tokenizer`, `load_gguf_bpe_tokenizer` |
| Tokenizers | `BytePairEncoder`, `UnigramEncoder`, `Gpt2BpeEncoder` |
| Aliases | `load_checkpoint` (= `load`), `export_classifier_model`, `import_classifier_model`, `quantize_classifier_model` (same as non-`_model` names) |
| Resource | `limit`, `limit_percent` |
| Errors | `CPUError`, `CUDAError`, `DataMismatchError`, `DataMissingRowError`, `ModelMismatchError` |

Typing: the package ships `py.typed` and `_native.pyi` stubs, so IDEs autocomplete
every class, method, and keyword argument.

---

## Datasets

| Class | Purpose |
|-------|---------|
| `DatasetQA` | JSON / JSONL / Parquet QA rows |
| `DatasetCorpus` | Two-file corpus (`rowfile` + `txtfile`, complexity sort) |
| `DatasetClassification` | Text + tags (auto `class_N` if tag column missing) |
| `DatasetImageGen` | Prompt + image + optional `negative_prompt` |
| `DatasetImageEdit` | Prompt + mask + image + optional `negative_prompt` |

### Common attributes

All dataset types expose:

- `rows` — sample count
- `format` — detected format string (`json`, `jsonl`, `corpus`, …)
- `type_` — logical type (`qa`, `corpus`, `classification`, `image_gen`, `image_edit`)

### DatasetQA

```python
ds = ai.DatasetQA(
    file="qa.json",           # or .jsonl / .parquet
    user_row="input",
    ai_row="output",
    system_row="systemprompt",  # optional
    thinktag="think|/think",    # open|close for CoT wrapping
    cot=True,
)
text = ds.format_sample(0)   # ChatXML conversation string
```

In-memory rows (no file needed; keys follow `user_row` / `ai_row`):

```python
ds = ai.DatasetQA(data=[
    {"input": "hi", "output": "hello"},
    {"input": "bye", "output": "goodbye"},
])
ds.format == "memory"
```

- Missing `user_row` / `ai_row` → `DataMissingRowError`
- Passing both `file=` and `data=` (or neither) → `ValueError`
- `repr()` shows row count and format

### DatasetCorpus

```python
ds = ai.DatasetCorpus(
    use_two_files=True,
    rowfile="rows.json",
    txtfile="corpus.txt",
    sort_rows_by_complexity=True,
    rows_with_corpus_chunk="text",
    batch_size="row",         # or fixed size string e.g. "24"
)
ds = ai.DatasetCorpus(data=["a chunk of text", "another chunk"])  # in-memory
ds.corpus_batch_size          # "row" or fixed integer string
```

### DatasetClassification

```python
ds = ai.DatasetClassification("labels.json", "text", "tag")
ds = ai.DatasetClassification(data=[{"text": "great", "label": "positive"}])
ds.unique_labels()            # sorted, deduplicated
```

Default columns are `text_col="text"` / `tags_col="label"`; in-memory `data=`
rows use the same keys.

### Image datasets

```python
gen = ai.DatasetImageGen("manifest.json")
edit = ai.DatasetImageEdit("edit_manifest.json")
gen.resolve_image_path("samples/cat.png")  # absolute path
gen.image_path_at(0)                       # first row image
gen.prompt_at(0)
edit.mask_path_at(0)
```

---

## Models

### Chatbot

```python
bot = ai.Chatbot(
    vocab_size=32000,
    n_layer=4,
    d_model=128,
    n_heads=4,                 # optional; default 4 when not using autoset
    n_kv_heads=2,              # optional grouped-query attention (default = n_heads)
    vision=False,
    autoset=None,              # or "sub-100M" | "sub-1B" | "sub-10B"
    seed=42,                   # optional deterministic init
    use_learned_pos_embed=False,  # default: fixed sinusoidal PE at runtime
    max_seq_len=512,           # learned PE table rows when use_learned_pos_embed=True
)
```

**Position encoding**

| Mode | Flag | Checkpoint | Trained by `Train()` |
|------|------|------------|----------------------|
| Sinusoidal (default) | `use_learned_pos_embed=False` | none | no |
| Learned table | `use_learned_pos_embed=True` | safetensors key `pos_embed` | yes |

- Sinusoidal PE is applied at forward time (no extra checkpoint keys).
- Learned `pos_embed` is `[max_seq_len, d_model]`; sequences longer than `max_seq_len` raise a shape error.
- `export(..., "bin")` stores `use_learned_pos_embed` / `max_seq_len` in the architecture stub (weights are not saved in `bin`).
- RL updates `lm_head` only; SPIN runs `Train()` and can update learned PE. See [position_encoding_coverage.md](position_encoding_coverage.md).
- Runnable roundtrip: `python examples/learned_pos_embed_roundtrip.py`

**Validation:** unknown `autoset` presets, `vocab_size=0`, and setting both
`use_learned_pos_embed=True` and `use_rope=True` raise `ValueError` at
construction time.

**Getters:** `vocab_size`, `n_layer`, `d_model`, `n_heads`, `n_kv_heads`, `parameters`, `layer_size`, `tokenizer`, `has_vision`, `init_seed`, `uses_causal_attention`, `use_learned_pos_embed`, `max_seq_len`

**Core methods:**

- `train(dataset, config=None, *, epochs=None, batch_size=None, learning_rate=None, optimizer=None, cuda=None, verbose=None, bpe_encoder=None, unigram_encoder=None) -> list[float]` — accepts `DatasetQA` or `DatasetCorpus`; keyword overrides win over `config`; returns per-epoch mean losses
- `chat(prompt, *, max_new_tokens=64, temperature=0.8, top_p=0.95, top_k=0, repetition_penalty=1.1, stop_strings=None, bpe_encoder=None, unigram_encoder=None) -> str` — generation with beginner-friendly sampling defaults
- `save(path, format="safetensors", bpe_encoder=None, unigram_encoder=None)` — same engine as `ai.export`
- `Chatbot.load(path, format=None)` — static; auto-detects JSON/binary/bin formats and errors clearly on non-chatbot checkpoints

**Loss / generation methods:**

- `compute_loss(input_str, target_str, bpe_encoder=None, ...) -> float` — same tokenization as `Train`
- `compute_mean_loss(dataset_qa | dataset_corpus, bpe_encoder=None) -> float`
- `generate(prompt, max_new_tokens=32, temperature=0.0, top_k=0, top_p=0.0, min_p=0.0, repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0, use_kv_cache=True, bpe_encoder=None, unigram_encoder=None, image_patch=None, image_patches=None) -> str`
- `generate_tokens(...)` — same sampling kwargs; returns new token ids only
- `stop_token_ids` / `stop_strings` optional on both (generation halts early)

### Classifier

```python
clf = ai.Classifier(num_labels=3, input_dim=64)
clf = ai.Classifier.with_labels(["A", "B"], input_dim=64, seed=1)
clf = ai.Classifier.from_classification(ds, input_dim=64, seed=1)
probs = clf.predict("some text")        # dict label -> probability
best = clf.predict_label("some text")   # argmax label string
```

**Getters:** `labels`, `num_labels`, `input_dim`, `init_seed`

**Methods:**

- `train(dataset, config=None, *, epochs=None, batch_size=None, learning_rate=None, optimizer=None, cuda=None, verbose=None) -> list[float]`
- `save(path, format="safetensors")` / `Classifier.load(path, format=None)`
- `predict(text)`, `predict_label(text)`
- `compute_loss(text, label)`, `compute_mean_loss(dataset_classification)`

### Diffusion

```python
diff = ai.Diffusion()
diff.latent_channels   # getter
diff.parameters        # VAE + UNet conv weight count
diff.train(dataset_image_gen, epochs=2)          # -> list of epoch losses
diff.save("diff.mmn"); d2 = ai.Diffusion.load("diff.mmn")
diff.compute_mean_denoise_loss(dataset_image_gen, t=7)
diff.denoise_loss_on_image("photo.png", t=5)
diff.sample_rgb_patch(steps=8, seed=42)
```

Foundation VAE/UNet — see [limitations.md](limitations.md).

---

## Training

The method style (`bot.train(...)`, `clf.train(...)`, `diff.train(...)`) and the
function style (`ai.Train(...)`, ...) run the same engine; every training call
returns **one mean-loss value per epoch**.

```python
cfg = ai.TrainConfig(
    epochs=3,
    batch_size=8,              # accumulates micro-batches before optimizer step
    cuda=False,                # True requires CUDA build + GPU
    optimizer="hybrid",        # "adamw" | "muon" | "hybrid" (Muon+AdamW)
    learning_rate=3e-4,
    verbose=False,             # True prints "[magicmindnet] epoch i/n - mean loss ..."
)
```

All fields are readable/writable on `cfg`. `repr(cfg)` summarizes settings.
Unknown optimizer names raise `ValueError` (at construction and at train time).
`"muon"` routes matrix weights through Muon with AdamW for vectors — the same
hybrid path, named for discoverability.

```python
losses = ai.Train(chatbot, dataset_qa, cfg)              # list[float], one per epoch
losses = ai.Train(chatbot, dataset_qa, cfg, bpe_encoder=bpe)  # optional BytePairEncoder
losses = ai.Train(chatbot, dataset_qa, cfg, unigram_encoder=uni)  # optional UnigramEncoder (not both)
losses = ai.TrainClassifier(classifier, dataset_cls, cfg)
losses = ai.TrainDiffusion(diffusion, dataset_image_gen, cfg)
ai.RL(chatbot, dataset_qa, cfg, reward_amount=1.0, punishment_amount=0.5, rl_type="policy", bpe_encoder=bpe)
ai.SPIN(chatbot, selfplay_epochs=2, dataset=dataset_qa, bpe_encoder=bpe)
```

### BytePairEncoder

```python
bpe = ai.BytePairEncoder.train(["hello world", "hello there"], vocab_size=512, num_merges=32)
bpe = ai.BytePairEncoder.train_from_qa(dataset_qa, vocab_size=512, num_merges=32)
bpe = ai.BytePairEncoder.train_from_corpus(dataset_corpus, vocab_size=512, num_merges=32)
ids = bpe.encode("hello world")  # merge-aware token ids (clamped to vocab_size)
```

Pass `bpe_encoder=bpe` to `Train()` for BPE tokenization during QA and corpus LM training (max 32 tokens per sequence).

Persist merge rules with `bpe.save("tokenizer.mmn")` and `BytePairEncoder.load("tokenizer.mmn")` (`mmn-bpe-v1` JSON). See [checkpoints.md](checkpoints.md).

### UnigramEncoder

```python
uni = ai.UnigramEncoder.train(["hello world", "hello there"], vocab_size=512)
uni = ai.UnigramEncoder.train_from_qa(dataset_qa, vocab_size=512)
uni = ai.UnigramEncoder.train_from_corpus(dataset_corpus, vocab_size=512)
ids = uni.encode("hello world")  # Viterbi segmentation by piece log-probs
text = uni.decode(ids)
```

Pass `unigram_encoder=uni` to `Train()` / `RL` / `SPIN` / `compute_mean_loss` (do not pass both `bpe_encoder` and `unigram_encoder`).

Persist with `uni.save("tokenizer.mmn")` and `UnigramEncoder.load("tokenizer.mmn")` (`mmn-unigram-v1` JSON).

| API | Required dataset |
|-----|------------------|
| `Train`, `Chatbot.compute_mean_loss` | `DatasetQA` or `DatasetCorpus` |
| `RL`, `SPIN` | `DatasetQA` |
| `TrainClassifier`, `Classifier.compute_mean_loss` | `DatasetClassification` |
| `TrainDiffusion` | `DatasetImageGen` or `DatasetImageEdit` |

Wrong dataset type → `DataMismatchError`. See [training_coverage.md](training_coverage.md).

### Diffusion

```python
d = ai.Diffusion()
ai.TrainDiffusion(d, dataset_image_gen, cfg)
patch = d.sample_rgb_patch(steps=8, seed=42)  # 192 floats, 8×8×3 RGB
inpaint = d.sample_inpaint_rgb_patch("photo.png", "mask.png", steps=8, seed=42)
loss = d.denoise_loss_on_image("photo.png", t=5)
loss_masked = d.denoise_loss_on_image_masked("photo.png", "mask.png", t=5)
mean_loss = d.compute_mean_denoise_loss(dataset_image_gen, t=7)
d.sample_rgb_patch_to_png("out.png", steps=8, seed=42)
ai.export_diffusion(d, "safetensors", "diffusion.mmn")
d2 = ai.import_diffusion("safetensors", ["diffusion.mmn"])
ai.quantize_diffusion(d2, "int8")
```

See `examples/diffusion_train.py`, `examples/diffusion_edit_train.py`, `examples/diffusion_inpaint_sample.py`, `examples/diffusion_sample.py`, `examples/diffusion_roundtrip.py`.

---

## Checkpoints & merge

**Universal loader** — detects the model family (Chatbot / Classifier /
Diffusion) and the format (JSON, binary HF safetensors, bin stub, GGUF,
PyTorch `.pt`, NumPy `.npz`) from the file contents:

```python
model = ai.load("anything.mmn")   # or .safetensors / .gguf / .pt / .npz
```

Model-specific loaders (`Chatbot.load`, `Classifier.load`, `Diffusion.load`)
raise `ValueError` naming the actual family when handed the wrong file.

| Function | Format | Notes |
|----------|--------|-------|
| `load(path)` | any | Auto-detects family + format; returns the right model type |
| `export(bot, "safetensors", path)` | `mmn-safetensors-v1` | Full weights + meta (JSON) |
| `export(bot, "hf-safetensors", path)` | `mmn-hf-safetensors-v1` | Full weights + meta (binary HF safetensors) |
| `export(bot, "safetensors", path, bpe_encoder=enc)` | `mmn-safetensors-v1` + `*.bpe.mmn` | Weights + `meta.bpe_checkpoint` sidecar |
| `export(bot, "safetensors", path, unigram_encoder=enc)` | `mmn-safetensors-v1` + `*.unigram.mmn` | Weights + `meta.unigram_checkpoint` sidecar |
| `load_bpe_sidecar(checkpoint_path)` | — | Load `mmn-bpe-v1` sibling referenced in meta |
| `load_unigram_sidecar(checkpoint_path)` | — | Load `mmn-unigram-v1` sibling referenced in meta |
| `export(bot, "bin", path)` | `mmn-bin-v1` | Architecture meta only |
| `export(bot, "gguf", path)` | GGUF v3 (F32) | From-scratch container; llama.cpp tensor names; `unigram_encoder=` embeds the vocab |
| `export(bot, "gguf-f16" \| "gguf-q8_0" \| "gguf-q4_0", path)` | GGUF v3 | Half-precision / block-quantized weights |
| `export(bot, "npz", path)` | NumPy `.npz` | `numpy.load`-compatible; `meta.json` entry |
| `export(bot, "pt", path)` | PyTorch state dict | `torch.load`-compatible; `_mmn_meta` entry |
| `import_model("safetensors", [path])` | JSON or binary | **First path only**; auto-detects HF binary; strict tensor validation |
| `import_model("hf-safetensors", [path])` | `mmn-hf-safetensors-v1` | Binary HF safetensors only |
| `import_model("gguf", [path])` | GGUF v2/v3 | Parallel dequant of **every current GGML type**: classic quants, Q2_K–Q8_K, IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS, TQ1_0/TQ2_0, MXFP4, NVFP4, F16/BF16 — cross-validated against llama.cpp's `gguf` package |
| `import_model("npz", [path])` | NumPy `.npz` | MMN or HF tensor names; stored or deflate entries |
| `import_model("pt", [path])` | PyTorch `.pt`/`.pth` | From-scratch pickle VM; zip and legacy pre-1.6 formats; HF llama-style state dicts adapt |
| `import_model("sharded", [index])` | HF `*.index.json` | `weight_map` shards (safetensors or torch), resolved next to the index |
| `export_classifier(clf, "safetensors", path)` | `mmn-classifier-v1` | backbone + head (JSON) |
| `export_classifier(clf, "hf-safetensors", path)` | `mmn-hf-classifier-v1` | backbone + head (binary HF) |
| `import_classifier("safetensors", [path])` | JSON or binary | **First path only**; auto-detects HF binary |
| `import_classifier("hf-safetensors", [path])` | `mmn-hf-classifier-v1` | Binary HF classifier only |
| `merge(a, b)` | — | Average Chatbot weights; shape must match |
| `merge_classifier(a, b)` | — | Labels + `input_dim` must match |
| `export_diffusion(d, "safetensors", path)` | `mmn-diffusion-v1` | VAE enc/dec + UNet conv weights |
| `import_diffusion("safetensors", [path])` | `mmn-diffusion-v1` | **First path only** |
| `merge_diffusion(a, b)` | — | Average VAE + UNet; `latent_channels` must match |
| `quantize_diffusion(d, "int8" \| "int4")` | — | In-place VAE + UNet conv weights |
| `quantize(model, "int8" \| "int4")` | — | In-place Chatbot weights |
| `quantize_classifier(clf, "int8" \| "int4")` | — | In-place Classifier weights |

Cross-import (chatbot ↔ classifier) is rejected. Full IO matrix: [checkpoint_coverage.md](checkpoint_coverage.md).

`init_seed` in meta when set at construction.

---

## Utilities

```python
ai.limit("50%")       # 1–100; also accepts "50" without %
pct = ai.limit_percent()
```

### Array file IO (NumPy / PyTorch interchange)

No numpy or torch installation needed; objects with `.tolist()` (numpy
arrays, torch tensors) are accepted anywhere an array is:

```python
ai.save_npy("x.npy", [[1.0, 2.0]])
x = ai.load_npy("x.npy")                       # nested lists

ai.save_npz("many.npz", {"w": [[1.0]], "b": [0.5]})
arrays = ai.load_npz("many.npz")               # {name: nested lists}

ai.save_pt("state.pt", {"w": [[1.0]]})         # torch.load-compatible
tensors = ai.load_pt("state.pt")               # zip + legacy pre-1.6 formats

weights = ai.load_h5("model.weights.h5")       # HDF5 without h5py
weights = ai.load_keras("model.keras")         # Keras v3 archive

info = ai.gguf_info("model.gguf")              # header-only inspection
tok = ai.load_gguf_tokenizer("model.gguf")     # embedded SentencePiece vocab
bpe = ai.load_gguf_bpe_tokenizer("llama3.gguf")  # gpt2-style byte-level BPE
ai.save_npz("small.npz", arrays, compress=True)  # from-scratch DEFLATE
bot = ai.load("pytorch_model.bin.index.json")  # sharded HF checkpoints
```

Details: [interop.md](interop.md).

---

## Errors

All subclass `Exception` with `message`, `fix`, and `explanation` fields where applicable:

| Type | When |
|------|------|
| `CPUError` | CPU backend unavailable |
| `CUDAError` | CUDA requested but not available |
| `DataMismatchError` | Dataset type wrong for model/API |
| `DataMissingRowError` | Required column missing in data file |
| `ModelMismatchError` | Merge on incompatible architectures |

---

## Examples

| Script | Command |
|--------|---------|
| Quickstart | `python examples/quickstart.py` |
| Train benchmark | `python examples/benchmark_train.py` (optional `--learned-pe`) |
| RL + SPIN | `python examples/rl_spin.py` |
| Mean loss | `python examples/eval_mean_loss.py qa`, `cls`, or `corpus` (optional `--train`, `--learned-pe`) |
| Classification | `python examples/classification.py` |
| Roundtrips | `python examples/checkpoint_roundtrip.py` |
| Learned PE roundtrip | `python examples/learned_pos_embed_roundtrip.py` |

Full list: [examples/README.md](../examples/README.md). CI smoke: `.\scripts\smoke_examples.ps1`.
