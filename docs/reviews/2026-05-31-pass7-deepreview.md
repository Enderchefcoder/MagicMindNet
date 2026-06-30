# Pass 7 artifact

- `TrainClassifier` / `Classifier::train_step` with full CE → head → GELU → backbone backward
- `validate_dataset_for_classifier`, `tests/test_train_classifier.py`
- Verified: 29 Rust unit tests, 26 pytest
