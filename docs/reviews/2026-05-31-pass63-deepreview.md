# Deep review pass 63 — 2026-05-31

## Scope

LayerNorm training scope, mmn-nn attention unit tests, CONTRIBUTING coverage links.

## Changes

- **`docs/layernorm_coverage.md`** — forward, frozen γ/β in LM training, IO/quantize, roadmap
- **`docs/nn_coverage.md`** — `mmn-nn` test matrix (LN + attention + block)
- **`train_step_does_not_update_layernorm_params`** — Rust regression in `mmn-models`
- **`mmn-nn/attention_tests`** — +4 tests (shape, K/V unused, block forward, FFN cache shapes)
- **`CONTRIBUTING.md`** — coverage matrix table linking all `docs/*_coverage.md` files

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 172
pytest: 411 passed
ruff: clean
```

## Merge-ready

YES

## Next (pass 64)

- `docs/testing.md` index of coverage docs
- Python test mirroring attn/LN frozen behavior (optional)
- AGENTS.md pointer to CONTRIBUTING coverage table
