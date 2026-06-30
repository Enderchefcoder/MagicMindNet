# Contributing to MagicMindNet

## Prerequisites

- Rust stable (`rustup`)
- Python 3.12+
- `pip install maturin pytest` or `pip install -e ".[dev]"` (includes ruff)

## Local loop

```powershell
cd MagicMindNet
python -m venv .venv
.\.venv\Scripts\Activate.ps1
maturin develop --release
cargo test --workspace
pytest
python examples/quickstart.py
```

Or run the full local CI script (`.venv` activation optional — scripts resolve venv Python automatically):

```powershell
.\scripts\ci_local.ps1
.\scripts\count_tests.ps1
.\scripts\smoke_examples.ps1
```

`scripts/venv_python.ps1` (Windows) and `scripts/venv_python.sh` (Unix) print the Python executable used by the gate. When `.venv` exists, CI uses it even if you forgot to `Activate.ps1`.

Merge gate (CI + counts in one step):

```powershell
.\scripts\verify_gate.ps1
```

On Linux/macOS:

```bash
bash scripts/ci_local.sh
bash scripts/count_tests.sh
bash scripts/verify_gate.sh
```

Python lint only:

```powershell
.\scripts\lint.ps1
```

CUDA (optional):

```powershell
maturin develop --release --features cuda -m crates/mmn-py/Cargo.toml
```

## TDD expectation

- Bug fixes and behavior changes: failing test first, then minimal fix.
- Run both `cargo test --workspace` and `pytest` before claiming done.

## Coverage matrices

Master index: [docs/testing.md](docs/testing.md). When adding features, extend the relevant regression doc and tests:

| Area | Doc |
|------|-----|
| Checkpoint IO | [docs/checkpoint_coverage.md](docs/checkpoint_coverage.md) |
| Training | [docs/training_coverage.md](docs/training_coverage.md) |
| Datasets | [docs/dataset_coverage.md](docs/dataset_coverage.md) |
| Examples smoke | [docs/examples_coverage.md](docs/examples_coverage.md) |
| Optimizers | [docs/optimizers_coverage.md](docs/optimizers_coverage.md) |
| Attention (alpha) | [docs/attention_coverage.md](docs/attention_coverage.md) |
| LayerNorm (alpha) | [docs/layernorm_coverage.md](docs/layernorm_coverage.md) |
| `mmn-nn` blocks | [docs/nn_coverage.md](docs/nn_coverage.md) |
| Classifier | [docs/classifier_coverage.md](docs/classifier_coverage.md) |
| Quantize | [docs/quantize_coverage.md](docs/quantize_coverage.md) |
| Known gaps / roadmap | [docs/limitations.md](docs/limitations.md) |
| PyO3 bindings layout | [docs/mmn_py_coverage.md](docs/mmn_py_coverage.md) |

## Checkpoint format

`mmn-safetensors-v1` stores `meta` plus tensors: `embed`, `lm_head`, and per-block linear weights (`blocks.N.attn.*`, `blocks.N.ffn`, `blocks.N.ffn2`). See `crates/mmn-io` tests for roundtrip guarantees.

## Pre-commit (optional)

```powershell
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Runs ruff via `.pre-commit-config.yaml`.

## Agents

See [AGENTS.md](AGENTS.md) and `.cursor/agents/` for Cursor subagent workflows.
