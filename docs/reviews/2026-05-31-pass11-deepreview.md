# Pass 11 — Deep review artifact

## Changes

- Python: `Chatbot.compute_loss`, `Classifier.compute_loss`, `DatasetClassification.unique_labels`
- PyO3: `RL` / `SPIN` `DataMismatchError` for non-QA datasets
- Tests: `test_chatbot_loss`, `test_classifier_loss`, `test_dataset_labels`, RL/SPIN mismatch
- Docs: API.md, limitations (`batch_size` note), benchmark_train loss printout

## Verification

- `cargo test --workspace`: 33 Rust tests, exit 0
- `pytest -q`: 35 passed
- `python examples/benchmark_train.py`: prints loss before/after, exit 0
