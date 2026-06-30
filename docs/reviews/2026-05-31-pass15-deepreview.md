# Pass 15 — Deep review artifact

## Changes

- `mean_classification_loss` + `Classifier.compute_mean_loss(DatasetClassification)`
- `TrainConfig` `#[pyo3(get, set)]` fields (writable from Python)
- Tests: `test_classifier_mean_loss.py`, `test_train_config_setters.py`

## Verification

- `cargo test --workspace` — 42 Rust unit tests
- `pytest -q` — 49 passed
- `scripts/ci_local.ps1` — see CI run output
