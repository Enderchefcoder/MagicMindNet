# Pass 17 — Deep review artifact

## Changes

- `mmn_optim::GradAccumulator` — average grads over `batch_size` micro-batches
- `Chatbot::backward_lm_grads` + accumulate/apply paths in `train()`
- `tests/test_train_batch_size.py`

## Verification

- `cargo test --workspace` — 44 Rust unit tests
- `pytest -q` — 52 passed
- `scripts/ci_local.ps1` — CI local: OK
