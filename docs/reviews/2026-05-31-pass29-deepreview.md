# Deep review pass 29 — 2026-05-31

## Changes

1. **Chatbot shape getters** — `vocab_size`, `n_layer`, `d_model` (PyO3).
2. **Classifier `num_labels` getter** — `len(labels)`.
3. **Tests** — shape getters, `num_labels`, merge `input_dim` mismatch (`ModelMismatchError`), `import_model("bin", [])`.
4. **Docs** — `docs/API.md`, `docs/testing.md`, `CHANGELOG.md`.

## Verification

- `cargo test --workspace`: 50 passed
- `pytest -q`: 98 passed
- `.\scripts\ci_local.ps1`: `CI local: OK`

## Merge-ready

YES (in-repo).
