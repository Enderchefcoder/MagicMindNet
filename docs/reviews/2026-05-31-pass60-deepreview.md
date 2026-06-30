# Deep review pass 60 — 2026-05-31

## Scope

`eval_mean_loss` corpus mode, mmn-io module split, conftest example harness.

## Changes

- **`examples/eval_mean_loss.py`** — `corpus` mode using fixture `DatasetCorpus`; smoke scripts updated
- **`tests/conftest.py`** — `run_example(script, *args)` supports CLI args
- **`tests/test_examples_scripts_py.py`** — smokes for `eval_mean_loss` (qa/cls/corpus) + `vision_chatbot`
- **`mmn-io/tensor_merge.rs`** — shared `average_tensors` + unit test; `merge_models` uses it
- **`mmn-io/classifier_io.rs`** — export/import/merge/quantize classifier extracted from `lib.rs`

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 165
pytest: 405 passed
ruff: clean
```

## Merge-ready

YES

## Next (pass 61)

- Extract chatbot safetensors export/import to `chatbot_io.rs`
- `eval_mean_loss` train-before/after optional flag
- GHA smoke for corpus eval_mean_loss if not already wired
