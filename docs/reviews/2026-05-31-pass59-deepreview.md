# Deep review pass 59 — 2026-05-31

## Scope

Vision chatbot metadata path, LayerNorm quantize coverage gaps, training docs polish.

## Changes

- **Vision path** — `vision_chatbot_trains_and_keeps_vision_flag` (Rust); `test_vision_chatbot_py.py`; `examples/vision_chatbot.py`; `docs/vision_coverage.md`
- **Quantize LN gaps** — non-default γ/β mutation tests (Rust + `test_io_ln_quantize_py.py`); `quantize_preserves_vision_flag_on_export_roundtrip`; `docs/quantize_coverage.md`
- **RL** — `punish_only` pytest in `test_train_rl_spin_py.py`
- **Training docs** — `training.md` RL mode table, corpus/vision/quantize cross-links; `training_coverage.md` vision row; `limitations.md` vision note; `checkpoint_coverage.md` LN footnote

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 164
pytest: 401 passed
ruff: clean
```

## Merge-ready

YES

## Next (pass 60)

- `eval_mean_loss` corpus mode example/test
- Further `mmn-io` split (merge helper, classifier module)
- Thermo deferred: split `mmn-io/src/lib.rs` (~1.6k lines)
