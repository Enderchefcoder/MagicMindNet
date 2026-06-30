# MagicMindNet — agent guide

## Build & test

```powershell
cd MagicMindNet
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
maturin develop --release
cargo test --workspace
pytest
.\scripts\ci_local.ps1
.\scripts\count_tests.ps1
.\scripts\verify_gate.ps1
```

Scripts under `scripts/` resolve `.venv` Python automatically when the venv exists (activation optional for `ci_local.ps1` / `smoke_examples.ps1`).

Linux/macOS:

```bash
bash scripts/ci_local.sh
bash scripts/verify_gate.sh
bash scripts/lint.sh
```

Test counts drift; use `count_tests.ps1` / `count_tests.sh` after adding tests (Rust `#[test]` + pytest collection).

CUDA: `maturin develop --release --features cuda -m crates/mmn-py/Cargo.toml`

## Layout

- `crates/mmn-core` — tensor, autograd, ops (CE grad, linear backward, embedding backward)
- `crates/mmn-optim` — AdamW, Muon, hybrid
- `crates/mmn-data` — datasets, ChatXML
- `crates/mmn-nn` — layers, VAE, UNet
- `crates/mmn-models` — Chatbot, Classifier, Diffusion
- `crates/mmn-train` — Train, RL, SPIN
- `crates/mmn-py` — PyO3 `_native` module ([split complete](docs/mmn_py_split_plan.md); `lib.rs` ~58 lines)
- `python/magicmindnet` — public Python API

## Conventions

- User-facing import: `import magicmindnet as ai`
- TDD: pytest + `cargo test` before claiming done
- No cyclic deps: `mmn-core` must not depend on `mmn-cuda`
- `Tensor.softmax(1)` on `[batch, classes]` normalizes each batch **row** (class dimension)

## Coverage matrices

When adding or changing behavior, extend the matching regression doc and tests. Full index: [docs/testing.md](docs/testing.md). Contributor table: [CONTRIBUTING.md](CONTRIBUTING.md#coverage-matrices).

Key docs: `checkpoint_coverage.md`, `training_coverage.md`, `examples_coverage.md`, `attention_coverage.md`, `layernorm_coverage.md`, `nn_coverage.md`.

## Subagents

- `.cursor/agents/magicmindnet-builder.md` — implementation
- `.cursor/agents/magicmindnet-reviewer.md` — review
- `.cursor/agents/magicmindnet-io.md` — export/import checkpoints
- `.cursor/agents/magicmindnet-train.md` — Train / RL / SPIN / loss tests
- `.cursor/agents/magicmindnet-classify.md` — DatasetClassification, TrainClassifier, labels
- `.cursor/agents/magicmindnet-python.md` — PyO3 bindings, pytest, examples
- `.cursor/agents/magicmindnet-ci.md` — `ci_local.ps1`, ruff, GitHub Actions
- `.cursor/agents/magicmindnet-docs.md` — README, `docs/`, CHANGELOG, examples
- `.cursor/agents/magicmindnet-examples.md` — `examples/` runnable demos
- `.cursor/agents/magicmindnet-gate.md` — autonomous deepreview / `ci_local` merge gate
