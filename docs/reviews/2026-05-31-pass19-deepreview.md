# Pass 19 — deep review artifact

- `TrainClassifier` + `TrainConfig.batch_size` gradient accumulation (`GradAccumulator`)
- `Classifier::backward_classifier_grads`, `apply_accumulated_classifier_grads`
- Tests: `tests/test_train_classifier_batch_size.py`, Rust `train_classifier_batch_size_two_accumulates_and_reduces_loss`
- GitHub Actions `ruff check`; docs/training/limitations/testing/AGENTS/CHANGELOG

Verification: `cargo test --workspace` 45 passed; `pytest -q` 55 passed; `ci_local.ps1` OK.
