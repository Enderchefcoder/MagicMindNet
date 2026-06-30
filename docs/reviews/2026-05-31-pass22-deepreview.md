# Pass 22 — deep review artifact

- `merge()` keeps first `Chatbot.init_seed`; Chatbot `__repr__` includes seed when set
- `DatasetCorpus.__repr__`; README Tests section
- Tests: `test_merge_chatbot_seed.py`, `test_dataset_corpus_repr.py`, Rust `merge_models_preserves_init_seed_from_first`

Verification: `cargo test --workspace` 48 passed; `pytest -q` 64 passed; `ci_local.ps1` OK.
