# Classifier coverage

Regression coverage for `Classifier`, `DatasetClassification`, and `TrainClassifier`.

## Text encoder (`encode_text`)

Alpha encoder maps raw bytes to a fixed-length vector for the linear backbone:

| Step | Behavior |
|------|----------|
| Input | UTF-8 `text.bytes()` |
| Length | First `input_dim` bytes; shorter texts zero-pad |
| Scale | Each byte → `(b as f32) / 255.0` ∈ [0, 1] |
| Forward | `gelu(backbone(x))` → `head` → softmax |

Implementation: `Classifier::encode_text` in `crates/mmn-models/src/chatbot.rs`.

Limitations: no n-grams, no subword tokenizer, no learned char-CNN — see [limitations.md](limitations.md).

## Prediction & loss

| Behavior | Test |
|----------|------|
| Encode byte / 255 | — | `test_classifier_edge_cases_py.py` (implicit via train) |
| `predict` probs sum ≈ 1 | `test_classifier_predict_probs.py` |
| Single-label predict | `test_classifier_edge_cases_py.py::test_single_label_classifier_predict_sums_to_one` |
| Empty text predict | `test_classifier_edge_cases_py.py::test_predict_empty_string_returns_all_labels` |
| Unknown label in `compute_loss` raises | `test_classifier_unknown_label.py` |
| `compute_mean_loss` skips unknown dataset tags | `test_classifier_edge_cases_py.py`, Rust `mean_classification_loss_skips_unknown_tags` |

## Training

| Behavior | Test |
|----------|------|
| `TrainClassifier` completes | `test_train_classifier.py` |
| Hybrid optimizer reduces mean loss | `test_classifier_edge_cases_py.py::test_hybrid_train_classifier_reduces_loss` |
| `batch_size=2` accumulation | `test_train_classifier_batch_size.py` |

## IO

Full matrix: [checkpoint_coverage.md](checkpoint_coverage.md) (classifier section).

## Examples

| Script | Smoke |
|--------|-------|
| `classification.py` | `test_examples_scripts_py.py`, `smoke_examples.ps1` / `.sh` |
| `classification_benchmark.py` | `test_examples_scripts_py.py`, `smoke_examples.ps1` |

## Running

```powershell
pytest tests/test_classifier_edge_cases_py.py -q
cargo test -p mmn-train mean_classification_loss
.\scripts\verify_gate.ps1
```

See [training.md](training.md) and [training_coverage.md](training_coverage.md).
