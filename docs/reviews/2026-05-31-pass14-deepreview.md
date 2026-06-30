# Pass 14 — Deep review artifact

## Changes

- `Classifier.with_labels_seed` / `from_classification_dataset_seed`
- Python `Classifier(..., seed=)`, `TrainConfig` getters
- `docs/testing.md`, `examples/classification_benchmark.py`
- Deduped autoset test (kept in `test_autoset.py`)

## Verification (2026-05-31)

- `cargo test --workspace` — 40 Rust unit tests, 0 failed
- `pytest -q` — 45 passed
- `python examples/classification_benchmark.py` — loss 0.6956 → 0.0000 (seed=42)
