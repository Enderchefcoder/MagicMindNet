# Deep review pass 24 — 2026-05-31

## Changes

- `Diffusion.__repr__`, `latent_channels` getter
- `DatasetImageGen` / `DatasetImageEdit` `format` getter; repr includes format
- Tests: diffusion repr, image format, RL CUDA, version/pyproject sync
- `examples/classifier_roundtrip.py`, `tests/fixtures/labels_small.json`
- `.cursor/agents/magicmindnet-examples.md`

## Verification

- `cargo test --workspace`: 48 Rust unit tests, 0 failed
- `pytest -q`: 72 passed
- `ruff check python tests examples`: clean
- `.\scripts\ci_local.ps1`: exit 0, `CI local: OK`
- `examples/classifier_roundtrip.py` / `checkpoint_roundtrip.py`: ok
