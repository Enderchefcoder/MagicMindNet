# Deep review pass 26 — 2026-05-31

## Changes

- Tests: `test_export_format_errors.py`, `test_dataset_qa_format_sample.py`, int4 quantize tests
- `pyproject.toml` `[dependency-groups]` dev
- `.github/workflows/ci.yml` roundtrip examples
- Docs: `docs/testing.md`, `docs/checkpoints.md`, `magicmindnet-io` agent

## Verification

- `cargo test --workspace`: 49 Rust tests, 0 failed
- `pytest -q`: 84 passed
- `ruff`: clean
- `.\scripts\ci_local.ps1`: exit 0
