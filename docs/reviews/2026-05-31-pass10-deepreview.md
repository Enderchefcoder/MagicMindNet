# Pass 10 — Deep review artifact

## Scope

- `mmn-core::ops::embedding_backward`
- `Chatbot::train_step_lm` embed optimizer step
- `Train()` PyO3 dataset guard
- Docs: `limitations.md`, `training.md`, `CHANGELOG`, `examples/benchmark_train.py`
- Tests: `embedding_backward_accumulates_rows`, `train_step_updates_embed_and_ffn2`, `test_train_rejects_classification_dataset`

## Verification (2026-05-31)

- `cargo test --workspace`: 33 unit tests, exit 0
- `pytest -q`: 30 passed
- `python examples/benchmark_train.py`: exit 0

## Merge-ready

YES — no open blockers in-repo.
