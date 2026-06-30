# Deep review pass 31 — 2026-05-31

## Changes

1. **Tests** — unknown quantize mode (chatbot + classifier); dataset `rows`/`type_`; merge/safetensors shape getter roundtrips.
2. **Scripts** — `scripts/verify_gate.ps1` wraps `ci_local` + `count_tests`.
3. **Docs** — `CONTRIBUTING.md`, `AGENTS.md`, `docs/testing.md`, `README.md`.

## Verification

- `.\scripts\verify_gate.ps1`: `verify_gate: OK`
- **50** Rust tests, **109** pytest

## Merge-ready

YES (in-repo).
