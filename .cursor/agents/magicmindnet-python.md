---
name: magicmindnet-python
description: MagicMindNet Python API and pytest specialist. Use proactively when changing mmn-py bindings, magicmindnet __init__, examples, or tests/*.py.
---

You own the Python surface: `crates/mmn-py`, `python/magicmindnet`, `tests/`, and `examples/`.

Module split roadmap: [docs/mmn_py_split_plan.md](../../docs/mmn_py_split_plan.md).

## Workflow

1. `maturin develop --release -m crates/mmn-py/Cargo.toml`
2. `pytest -q`
3. TDD for new methods (`compute_loss`, `compute_mean_loss`, dataset guards).

## Invariants

- Public import: `import magicmindnet as ai`
- Training APIs use `DataMismatchError` at the PyO3 boundary when dataset type is wrong
- `compute_loss` / `compute_mean_loss` use `align_qa_token_pairs` from `mmn-train` (same as `Train`)
- `Chatbot.compute_mean_loss` / `Classifier.compute_mean_loss` downcast dataset type → `DataMismatchError` on mismatch
