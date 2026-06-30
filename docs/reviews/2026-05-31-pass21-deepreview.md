# Pass 21 — deep review artifact

- `DatasetClassification.__repr__`; Classifier repr includes `init_seed` when set
- Export omits `meta.seed` when unset; `merge_classifier` preserves first model `init_seed`
- `.pre-commit-config.yaml`, `magicmindnet-docs` subagent, CONTRIBUTING pre-commit section

Verification: `cargo test --workspace` 47 passed; `pytest -q` 62 passed; `ruff check` clean; `ci_local.ps1` OK.
