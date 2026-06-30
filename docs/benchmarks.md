# Benchmarks (baseline)

Run locally after `maturin develop --release`:

| Command | Purpose |
|---------|---------|
| `cargo test -p mmn-core` | Tensor / autograd |
| `cargo test -p mmn-optim` | AdamW + Muon NS |
| `pytest tests/test_train.py` | End-to-end train smoke |

Record date, OS, and CPU/GPU in PRs when changing hot paths.
