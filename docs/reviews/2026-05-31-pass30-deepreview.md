# Deep review pass 30 — 2026-05-31

## Changes

1. **Tests** — bin shape/vision getters; classifier `input_dim`/`num_labels` roundtrip; autoset shape getters; `has_vision`; import missing file.
2. **CI** — GitHub Actions runs `examples/quickstart.py` (parity with `ci_local.ps1`).
3. **Docs** — `docs/checkpoints.md` getter examples; `docs/testing.md` inventory.
4. **Subagent** — `.cursor/agents/magicmindnet-gate.md` for autonomous merge gates.

## Verification

- `cargo test --workspace`: 50 passed
- `pytest -q`: 103 passed
- `.\scripts\ci_local.ps1`: `CI local: OK`

## Merge-ready

YES (in-repo).
