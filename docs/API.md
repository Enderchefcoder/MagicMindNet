# MagicMindNet API Reference

Import as:

```python
import magicmindnet as ai
```

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
| Training | `TrainConfig`, `Train`, `TrainClassifier`, `RL`, `SPIN` |
| IO | `export`, `import_model`, `merge`, `quantize`, `export_classifier`, `import_classifier`, `merge_classifier`, `quantize_classifier` |
| Aliases | `export_classifier_model`, `import_classifier_model`, `quantize_classifier_model` (same as non-`_model` names) |
| Resource | `limit`, `limit_percent` |
| Errors | `CPUError`, `CUDAError`, `DataMismatchError`, `DataMissingRowError`, `ModelMismatchError` |

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

- Missing `user_row` / `ai_row` → `DataMissingRowError`
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
ds.corpus_batch_size          # "row" or fixed integer string
```

### DatasetClassification

```python
ds = ai.DatasetClassification("labels.json", "text", "tag")
ds.unique_labels()            # sorted, deduplicated
```

### Image datasets

```python
gen = ai.DatasetImageGen("manifest.json")
edit = ai.DatasetImageEdit("edit_manifest.json")
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

**Getters:** `vocab_size`, `n_layer`, `d_model`, `n_heads`, `n_kv_heads`, `parameters`, `layer_size`, `tokenizer`, `has_vision`, `init_seed`, `uses_causal_attention`, `use_learned_pos_embed`, `max_seq_len`

**Methods:**

- `compute_loss(input_str, target_str) -> float`
- `compute_mean_loss(dataset_qa | dataset_corpus, bpe_encoder=None) -> float`
- `generate(prompt, max_new_tokens=32, temperature=0.0, top_k=0, bpe_encoder=None) -> str`
- `generate_tokens(...)` — same kwargs; returns new token ids only
- `stop_token_ids` / `stop_strings` optional on both (generation halts early)
- `compute_loss(input, target, bpe_encoder=None) -> float` — same tokenization as `Train`

### Classifier

```python
clf = ai.Classifier(num_labels=3, input_dim=64)
clf = ai.Classifier.with_labels(["A", "B"], input_dim=64, seed=1)
clf = ai.Classifier.from_classification(ds, input_dim=64, seed=1)
probs = clf.predict("some text")   # dict label -> probability
```

**Getters:** `labels`, `num_labels`, `input_dim`, `init_seed`

**Methods:** `compute_loss(text, label)`, `compute_mean_loss(dataset_classification)`

### Diffusion

```python
diff = ai.Diffusion()
diff.latent_channels   # getter
```

Foundation VAE/UNet — see [limitations.md](limitations.md).

---

## Training

```python
cfg = ai.TrainConfig(
    epochs=3,
    batch_size=8,              # accumulates micro-batches before optimizer step
    cuda=False,                # True requires CUDA build + GPU
    optimizer="hybrid",        # "hybrid" (Muon+AdamW) or "adamw"
    learning_rate=3e-4,
)
```

All fields are readable/writable on `cfg`. `repr(cfg)` summarizes settings.

```python
ai.Train(chatbot, dataset_qa, cfg)
ai.Train(chatbot, dataset_qa, cfg, bpe_encoder=bpe)  # optional BytePairEncoder
ai.TrainClassifier(classifier, dataset_cls, cfg)
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

| API | Required dataset |
|-----|------------------|
| `Train`, `Chatbot.compute_mean_loss` | `DatasetQA` or `DatasetCorpus` |
| `RL`, `SPIN` | `DatasetQA` |
| `TrainClassifier`, `Classifier.compute_mean_loss` | `DatasetClassification` |

Wrong dataset type → `DataMismatchError`. See [training_coverage.md](training_coverage.md).

---

## Checkpoints & merge

| Function | Format | Notes |
|----------|--------|-------|
| `export(bot, "safetensors", path)` | `mmn-safetensors-v1` | Full weights + meta (JSON) |
| `export(bot, "hf-safetensors", path)` | `mmn-hf-safetensors-v1` | Full weights + meta (binary HF safetensors) |
| `export(bot, "safetensors", path, bpe_encoder=enc)` | `mmn-safetensors-v1` + `*.bpe.mmn` | Weights + `meta.bpe_checkpoint` sidecar |
| `load_bpe_sidecar(checkpoint_path)` | — | Load `mmn-bpe-v1` sibling referenced in meta |
| `export(bot, "bin", path)` | `mmn-bin-v1` | Architecture meta only |
| `import_model("safetensors", [path])` | JSON or binary | **First path only**; auto-detects HF binary; strict tensor validation |
| `import_model("hf-safetensors", [path])` | `mmn-hf-safetensors-v1` | Binary HF safetensors only |
| `export_classifier(clf, "safetensors", path)` | `mmn-classifier-v1` | backbone + head (JSON) |
| `export_classifier(clf, "hf-safetensors", path)` | `mmn-hf-classifier-v1` | backbone + head (binary HF) |
| `import_classifier("safetensors", [path])` | JSON or binary | **First path only**; auto-detects HF binary |
| `import_classifier("hf-safetensors", [path])` | `mmn-hf-classifier-v1` | Binary HF classifier only |
| `merge(a, b)` | — | Average Chatbot weights; shape must match |
| `merge_classifier(a, b)` | — | Labels + `input_dim` must match |
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
