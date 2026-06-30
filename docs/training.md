# Training guide

## Language modeling (`Chatbot`)

```python
import magicmindnet as ai

ds = ai.DatasetQA(file="data/qa.json", user_row="input", ai_row="output")
bot = ai.Chatbot(vocab_size=32000, n_layer=4, d_model=128)
cfg = ai.TrainConfig(epochs=3, batch_size=8, cuda=False, optimizer="hybrid", learning_rate=3e-4)
ai.Train(bot, ds, cfg)  # batch_size>1 accumulates grads over N QA rows per optimizer step
```

Rust `train_step_lm` applies cross-entropy gradients to `lm_head`, **embedding**, optional **learned `pos_embed`**, and **every** block’s attn + FFN + LayerNorm γ/β (see [attention_coverage.md](attention_coverage.md)). Sinusoidal PE is fixed at runtime. Opt-in learned PE: `Chatbot(..., use_learned_pos_embed=True, max_seq_len=64)` — see [position_encoding_coverage.md](position_encoding_coverage.md).

`Train()` raises `DataMismatchError` if you pass a `DatasetClassification` instead of `DatasetQA` or `DatasetCorpus`.

Monitor training:

```python
print("mean CE:", bot.compute_mean_loss(ds))
```

## Classification

```python
ds = ai.DatasetClassification("labels.json", "text", "tag")
clf = ai.Classifier.from_classification(ds, input_dim=64)
cfg = ai.TrainConfig(epochs=5, batch_size=4, learning_rate=0.05)
ai.TrainClassifier(clf, ds, cfg)  # batch_size>1 accumulates over N labeled rows per step
print("mean CE:", clf.compute_mean_loss(ds))
probs = clf.predict("example text")  # dict label -> probability
ai.export_classifier(clf, "safetensors", "classifier.mmn")
clf2 = ai.import_classifier("safetensors", ["classifier.mmn"])
```

See `examples/classification.py` for a full script.

Full regression matrix: [training_coverage.md](training_coverage.md).

Classifier edge cases: [classifier_coverage.md](classifier_coverage.md).

Corpus LM training: `Train(chatbot, DatasetCorpus, …)` with next-token CE; `compute_mean_loss` accepts corpus too. Learned `pos_embed` trains on corpus rows the same as QA (`test_train_corpus_learned_pos_embed_reduces_mean_loss`).

Example scripts with `--learned-pe`: `benchmark_train.py`, `corpus_benchmark.py`, `eval_mean_loss.py` (QA/corpus). See [training_coverage.md](training_coverage.md).

Diffusion smoke: [diffusion_coverage.md](diffusion_coverage.md).

Optimizer / autograd unit coverage: [optimizers_coverage.md](optimizers_coverage.md).

Vision-flag chatbots (`vision=True`): metadata + IO path only — see [vision_coverage.md](vision_coverage.md).

Quantize regression matrix: [quantize_coverage.md](quantize_coverage.md).

Attention training scope (alpha): [attention_coverage.md](attention_coverage.md).

## RL / SPIN

```python
ai.RL(bot, ds, cfg, reward_amount=1.0, punishment_amount=0.5, rl_type="policy")
ai.SPIN(bot, selfplay_epochs=2, dataset=ds)
```

| `rl_type` | Behavior |
|-----------|----------|
| `policy` (default) | Reward rows use `reward_amount`; punished rows use `punishment_amount` |
| `reward_only` | Only rewarded rows update weights |
| `punish_only` | Only punished rows update weights |
| `selfplay` | Same as `reward_only` (heuristic self-play scale) |

## Checkpoints

```python
ai.export(bot, "safetensors", "checkpoint.mmn")
bot2 = ai.import_model("safetensors", ["checkpoint.mmn"])
```

Format: `mmn-safetensors-v1` (JSON wrapper with `meta` + tensors). See [limitations.md](limitations.md).

## Resource limit

```python
ai.limit("50%")  # 1–100; lowers process priority on Windows/Unix
```
