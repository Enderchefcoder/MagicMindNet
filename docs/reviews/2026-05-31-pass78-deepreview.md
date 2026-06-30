# Deep review pass 76–78 — 2026-05-31

## Scope

Complete `mmn-py` split plan steps 5–7: models, train/io glue, thin `lib.rs`.

## Changes

### Pass 76 — `models/chatbot.rs`
- `PyChatbot` with `pub(crate) inner`, getters, `compute_loss`, `compute_mean_loss`

### Pass 77 — `models/classifier.rs`
- `PyClassifier` with `from_classification`, `predict`, loss APIs

### Pass 78 — `train/mod.rs`, `io/mod.rs`, thin `lib.rs`
- **train/** — `Train`, `TrainClassifier`, `RL`, `SPIN`
- **io/** — `export`/`import`/`merge`/`quantize` (chatbot + classifier)
- **lib.rs** — `#[pymodule]` registration only (**58 lines**)

## Final `mmn-py` layout

```
crates/mmn-py/src/
  lib.rs           (58 lines — pymodule registry)
  errors.rs
  train_config.rs
  resource.rs
  datasets/        (qa, corpus, classification, image)
  models/          (chatbot, classifier, diffusion)
  train/           (Train, RL, SPIN, TrainClassifier)
  io/              (export, import, merge, quantize)
```

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 172
pytest: 422 passed
ruff: clean
```

## Merge-ready

YES — split plan acceptance met; Python API unchanged.

## Next (pass 79+)

- Coverage/docs polish for new module paths in `AGENTS.md`
- Attention backward / NN hardening per `limitations.md`
- Initial git commit / PR (NEEDS HUMAN)
