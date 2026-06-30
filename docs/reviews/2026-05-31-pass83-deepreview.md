# Deep review pass 83 — 2026-05-31

## Scope

Residual backward audit + LN γ/β finite-diff coverage + FFN residual fix.

## Changes

- **FFN residual** — `out = x2 + ffn(ln2(x2))` in `forward_with_cache` (was FFN-only output)
- **`backward_attn_ffn`** — `grad_x2` now starts from `grad_out` (FFN skip) before LN2 chain
- **Tests** — `layernorm_backward_gamma_beta_finite_diff`; `transformer_block_uses_ffn_residual`; `transformer_block_backward_input_matches_finite_diff`
- **Docs** — `nn_coverage.md`, `layernorm_coverage.md`, `limitations.md`, `attention_coverage.md` block-structure section

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 179 (+3)
pytest: 429 passed
ruff: clean
```

## Merge-ready

YES

## Next (pass 84)

- Rename `test_train_frozen_attn_ln_py.py` → `test_train_block_params_py.py` (filename now misleading)
- Optional: causal attention mask; cross-attn sketch for `vision=True`
