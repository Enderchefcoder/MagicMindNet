# Deep review pass 25 — 2026-05-31

## Changes

- `import_safetensors` format guard (`mmn-safetensors-v1` only)
- Tests: `test_import_format_guard.py`, `test_quantize_classifier.py`, `test_dataset_classification_unique_labels.py`
- `scripts/smoke_examples.ps1` + `ci_local.ps1` step
- Docs: `docs/checkpoints.md`, `CONTRIBUTING.md`, `CHANGELOG.md`

## Verification

- `cargo test --workspace`: 49 Rust `#[test]` (mmn-io 12), 0 failed
- `pytest -q`: 77 passed
- `ruff check python tests examples`: clean
- `.\scripts\ci_local.ps1`: exit 0, includes examples smoke
