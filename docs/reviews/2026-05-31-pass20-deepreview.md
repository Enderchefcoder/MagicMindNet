# Pass 20 — deep review artifact

- `init_seed` on `Chatbot` / `Classifier`; optional `meta.seed` in v1 checkpoints
- PyO3 getters `init_seed`; `DatasetQA.__repr__`
- Rust IO tests: `export_includes_seed_in_meta`, `classifier_export_includes_seed_in_meta`
- Python: `test_checkpoint_meta_seed.py`, `test_dataset_qa_repr.py`

Verification: `cargo test --workspace` 47 passed; `pytest -q` 59 passed; `ci_local.ps1` OK.
