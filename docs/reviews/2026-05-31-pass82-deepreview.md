# Deep review pass 82 — 2026-05-31

## Scope

LayerNorm γ/β backward wired into `train_step_lm` (completes block backward after pass 81 attn).

## Changes

- **`layernorm_row_backward` / `layernorm_backward`** in `mmn-nn` with finite-diff test (one-hot `grad_out`)
- **`BlockForwardCache`** stores `x_in`, `x2` for LN backward
- **`backward_attn_ffn`** returns 10 weight grads per block (adds ln2_γ/β, ln1_γ/β)
- **`apply_block_lm_grads`** helper in `chatbot.rs`; optimizer steps for all 10 block tensors
- **Tests flipped** — `train_step_updates_layernorm_gamma`; pytest `test_train_changes_ln1_gamma`, `test_spin_changes_ln1_gamma`; RL LN still frozen

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 176 (+1 layernorm backward test; row-mean test retained)
pytest: 429 passed
ruff: clean
```

## Merge-ready

YES

## Next (pass 83)

- Stale doc cleanup in `attention_coverage.md` (remove pre-pass-81 frozen-attn language)
- Optional: γ/β finite-diff unit test; residual-add backward audit in block forward path
