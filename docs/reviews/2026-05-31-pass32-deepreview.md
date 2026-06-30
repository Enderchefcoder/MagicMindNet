# Deep review pass 32 — 2026-05-31

## Changes

1. **Tests** — classifier `predict` probabilities sum to 1; `TrainConfig()` defaults; documented getters on Chatbot/Classifier; merge preserves labels/`input_dim`/`num_labels`; `limit("25")` without `%`.
2. **Python** — `__version__` added to `magicmindnet.__all__`.
3. **Docs** — `docs/API.md` `limit_percent`; `docs/testing.md` inventory.

## Verification

- `.\scripts\verify_gate.ps1`: `verify_gate: OK`
- **50** Rust, **116** pytest

## Merge-ready

YES (in-repo).
