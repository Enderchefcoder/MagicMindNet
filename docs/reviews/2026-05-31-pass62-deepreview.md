# Deep review pass 62 — 2026-05-31

## Scope

mmn-io test file split, examples coverage matrix, attention training documentation.

## Changes

- **`mmn-io/src/lib.rs`** — shrunk to re-exports only (~14 lines); regression tests moved to `io_tests/`
- **`io_tests/chatbot_io_tests.rs`** — 78 chatbot import/merge/quantize/bin tests
- **`io_tests/classifier_io_tests.rs`** — 20 classifier IO tests
- **`docs/examples_coverage.md`** — full examples × smoke × pytest matrix
- **`docs/attention_coverage.md`** — forward path, what trains, roadmap
- **`train_step_does_not_update_attn_weights`** — Rust regression for frozen attn in LM step
- **pytest** — smokes for `quickstart`, `checkpoint_roundtrip`, `classifier_roundtrip`

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 167
pytest: 411 passed
ruff: clean
```

## Merge-ready

YES

## Next (pass 63)

- LayerNorm γ/β training path or explicit doc test
- `mmn-nn` attention forward unit tests
- CONTRIBUTING link to coverage matrices
