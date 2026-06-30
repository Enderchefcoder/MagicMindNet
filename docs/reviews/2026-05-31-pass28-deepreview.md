# Deep review pass 28 — 2026-05-31

## Scope

Whole first-party tree; focus on Python API parity, smoke/CI coverage, docs.

## Changes

1. **Classifier `input_dim` getter** — PyO3 `#[getter]` exposing `inner.input_dim` (repr already showed it).
2. **Tests (TDD)** — `test_classifier_input_dim.py` (red→green), `test_chatbot_same_seed_loss.py`, `test_import_empty_files.py`.
3. **CI** — `eval_mean_loss.py cls` in `smoke_examples.ps1` and GitHub Actions python job.
4. **Docs** — `docs/API.md`, `README.md` test counts, `CHANGELOG.md` pass 28.

## Verification

- `cargo test --workspace`: 50 passed
- `pytest -q`: 94 passed
- `ruff check python tests examples`: clean
- `.\scripts\ci_local.ps1`: exit 0, `CI local: OK`

## Merge-ready

YES (in-repo); initial git commit/remote still NEEDS HUMAN.
