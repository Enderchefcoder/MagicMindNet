# Pass 12 — Deep review artifact

## Changes

- `align_qa_token_pairs` in `mmn-train` (train + `compute_loss` + `mean_qa_loss`)
- `Chatbot.compute_mean_loss(DatasetQA)` Python API
- Tests: `test_mean_qa_loss`, `test_export_loss`, Rust alignment/mismatch tests
- `scripts/ci_local.ps1`, `magicmindnet-python` agent

## Verification

- `cargo test --workspace`: 36 Rust unit tests, exit 0
- `pytest -q`: 38 passed
- `python examples/benchmark_train.py`: mean loss decreases on fixture

## Merge-ready

YES
