# Deep review pass 61 — 2026-05-31

## Scope

Chatbot IO module extraction, `eval_mean_loss --train`, GHA smoke documentation.

## Changes

- **`mmn-io/chatbot_io.rs`** — `export_safetensors`, `import_safetensors`, `merge_models`, `quantize_model`, `export_bin`, `import_bin`; embed/lm_head merge uses `average_tensors`; +1 roundtrip unit test
- **`examples/eval_mean_loss.py`** — optional `--train` prints before/after mean loss for qa, corpus, and cls
- **`tests/test_examples_scripts_py.py`** — smokes for `--train` on all three modes
- **CI** — GHA comment noting `smoke_examples.sh` covers `eval_mean_loss` qa|cls|corpus
- **Docs** — `checkpoints.md`, `API.md`, `examples/README.md`, `training_coverage.md`

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 166
pytest: 408 passed
ruff: clean
```

## Merge-ready

YES

## Next (pass 62)

- Move regression tests from `mmn-io/lib.rs` into per-module test files
- `docs/examples_coverage.md` matrix
- Attention backward stub or doc-only milestone note
