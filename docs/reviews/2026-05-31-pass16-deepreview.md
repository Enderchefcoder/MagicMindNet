# Pass 16 — Deep review artifact

## Changes

- `Chatbot.compute_mean_loss` → `Bound<PyAny>` + `DataMismatchError`
- `TrainConfig.__repr__`
- `train_classifier` shuffles sample order each epoch (matches `train`)
- `examples/eval_mean_loss.py`
- Removed spurious `mut` on `Classifier::with_labels`

## Verification

- `cargo test --workspace` — 42 Rust unit tests
- `pytest -q` — 51 passed
- `python examples/eval_mean_loss.py qa|cls`
