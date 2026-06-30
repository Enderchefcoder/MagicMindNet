# Deep review pass 55 — 2026-05-31

## Scope

Optimizer coverage, autograd/ops regression, Muon Newton-Schulz bugfix.

## Changes

- **Bugfix:** `newton_schulz5` used a wrong quintic iteration (zeroed all updates); replaced with Keller Jordan Muon coeffs `(3.4445, -4.775, 2.0315)` and `X = a*X + (b*A + c*A²)@X` where `A = X@Xᵀ`
- **Bugfix:** Muon Nesterov blend now matches reference `(1-β)*grad + β*momentum`
- `crates/mmn-optim` — +6 tests: `classify_param`, `grad_accumulator_clear`, `muon_step_matrix`, `hybrid_routes_matrix`, `newton_schulz_output_nonzero`
- `crates/mmn-core` — +3 tests: `backward_add_splits_grad_to_parents`, `linear_backward_grad_shapes`, `ce_grad_averages_over_batch`
- `docs/optimizers_coverage.md` — optimizer/autograd regression matrix
- `tests/test_optimizer_integration_py.py` — hybrid default + adamw Train smoke
- `docs/training.md` — link to optimizers coverage

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 150
pytest: 381 passed
ruff: clean
```

## Merge-ready

YES (library gate). Muon hybrid optimizer now actually updates 2D weight matrices.
