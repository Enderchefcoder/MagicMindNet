# Getting started with MagicMindNet

This tutorial takes you from zero to a trained, saved, reloaded chatbot and classifier. No machine-learning background needed.

## 1. Install

You need Python 3.12+ and [Rust](https://rustup.rs/) (the library core is compiled Rust).

```bash
git clone https://github.com/Enderchefcoder/MagicMindNet
cd MagicMindNet
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
maturin develop --release
```

Check it worked:

```bash
python -c "import magicmindnet as ai; print(ai.__version__)"
```

## 2. Your first chatbot

MagicMindNet trains real transformer language models from scratch — on your machine, on your data. Small models learn small things, and that is the point: you can watch every step.

```python
import magicmindnet as ai

# Training data is just a list of input/output pairs.
data = ai.DatasetQA(data=[
    {"input": "hi", "output": "hello there!"},
    {"input": "how are you?", "output": "doing great, thanks!"},
])

# A small transformer: 2 layers, 64-dimensional.
bot = ai.Chatbot(vocab_size=512, n_layer=2, d_model=64, seed=42)

# Train. Returns the mean loss of each epoch so you can see it learn.
losses = bot.train(data, epochs=5, verbose=True)

# Generate a reply.
print(bot.chat("hi"))
```

`verbose=True` prints one line per epoch:

```
[magicmindnet] epoch 1/5 - mean loss 6.2284
[magicmindnet] epoch 2/5 - mean loss 5.9268
...
```

Loss going down means the model is learning. A brand-new toy model produces mostly noise — train longer, add more data, or use a bigger model to improve it.

### Loading data from files

For real datasets, point `DatasetQA` at a JSON, JSONL, or Parquet file:

```python
data = ai.DatasetQA(file="qa.json", user_row="question", ai_row="answer")
```

where `qa.json` looks like `[{"question": "...", "answer": "..."}, ...]`.

### Model size presets

Don't want to pick `n_layer`/`d_model` yourself? Use a parameter-budget preset:

```python
bot = ai.Chatbot(autoset="sub-100M")   # also: "sub-1B", "sub-10B"
```

## 3. Save and load

```python
bot.save("my_bot.mmn")        # one file, safe to copy anywhere
bot = ai.load("my_bot.mmn")   # ai.load() figures out what's in the file
```

`ai.load()` works on every checkpoint MagicMindNet writes — chatbots, classifiers, and diffusion models — so you never have to remember which import function to use. (Model-specific loaders like `ai.Chatbot.load(path)` also exist and error clearly when handed the wrong file.)

Checkpoint loading is **strict**: a corrupt or truncated file fails loudly instead of silently mixing random weights.

## 4. Your first classifier

```python
labels = ai.DatasetClassification(data=[
    {"text": "sunny warm bright", "label": "nice"},
    {"text": "rainy cold windy", "label": "gloomy"},
])

clf = ai.Classifier.from_classification(labels, input_dim=64, seed=42)
clf.train(labels, epochs=20)

print(clf.predict_label("sunny warm bright"))   # -> "nice"
print(clf.predict("sunny warm bright"))         # -> {"nice": 0.93, "gloomy": 0.07}
```

Classifier checkpoints save and load the same way: `clf.save("clf.mmn")`, `ai.load("clf.mmn")`.

## 5. Tuning training

Every training call accepts the same options, either as keywords or a reusable `TrainConfig`:

```python
cfg = ai.TrainConfig(
    epochs=3,
    batch_size=8,          # gradient accumulation over this many samples
    learning_rate=3e-4,
    optimizer="hybrid",    # "adamw", "muon", or "hybrid" (Muon + AdamW)
    cuda=False,            # True needs a CUDA build
    verbose=False,
)
bot.train(data, cfg)
bot.train(data, cfg, epochs=10)   # keywords override the config
```

Typos are caught immediately: `optimizer="sgd"` raises `ValueError` listing the valid names.

## 6. Better text with a trained tokenizer

By default, text becomes tokens byte-by-byte. Training a tokenizer on your data usually improves results:

```python
bpe = ai.BytePairEncoder.train_from_qa(data, vocab_size=512, num_merges=32)
bot.train(data, epochs=5, bpe_encoder=bpe)
print(bot.chat("hi", bpe_encoder=bpe))

bot.save("bot.mmn", bpe_encoder=bpe)      # tokenizer saved beside the model
bpe = ai.load_bpe_sidecar("bot.mmn")      # ...and loaded back
```

## 7. Where to go next

| Topic | Doc |
|-------|-----|
| Full API reference | [API.md](API.md) |
| All runnable examples | [../examples/README.md](../examples/README.md) |
| Training internals (optimizers, losses, batching) | [training.md](training.md) |
| Checkpoint formats & merging models | [checkpoints.md](checkpoints.md) |
| Diffusion (image) models | [diffusion_coverage.md](diffusion_coverage.md) |
| Known limitations of the alpha | [limitations.md](limitations.md) |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: magicmindnet._native` | Run `maturin develop --release` inside the venv |
| `CUDAError` | You passed `cuda=True` without a CUDA build; rebuild with `maturin develop --release --features cuda` or use `cuda=False` |
| `DataMismatchError` | The dataset type doesn't match the model (e.g. classification data on a Chatbot); the message names the expected type |
| `DataMissingRowError` | A dataset row/column key is missing; the message names the missing key |
| Replies are gibberish | Normal for tiny fresh models — more data, more epochs, a tokenizer (`bpe_encoder=`), or a bigger model |
