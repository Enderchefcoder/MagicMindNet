---
name: magicmindnet-builder
description: Implements MagicMindNet milestones in Rust/PyO3 with TDD. Use for feature work, bugfixes, and running cargo test + pytest.
---

You build MagicMindNet (`import magicmindnet as ai`).

## Workflow

1. Read `.cursorrules` Scratchpad for current milestone.
2. Write failing `pytest` or `cargo test` first.
3. Implement in `crates/mmn-*` and `crates/mmn-py`.
4. Run `cargo test --workspace` and `pytest` before claiming done.
5. Keep files under 1k lines; split crates instead of growing monoliths.

## Conventions

- Public Python names match the design sketch: `DatasetQA`, `Train`, `RL`, `SPIN`, `merge`, `limit`, `export`, `import_model`, `quantize`.
- Errors: `CPUError`, `CUDAError`, `DataMismatchError`, `DataMissingRowError`, `ModelMismatchError` with detailed messages.
- Optimizers: hybrid = Muon on 2D weights, AdamW on the rest.
