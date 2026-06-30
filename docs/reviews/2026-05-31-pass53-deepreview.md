# Deep review pass 53 — 2026-05-31

## Scope

Dataset loader coverage across QA, corpus, classification, image, and ChatXML.

## Changes

- `docs/dataset_coverage.md` — loader regression matrix
- `tests/test_dataset_matrix_py.py` — jsonl, missing ai_row, auto-tags, image fields, type getters
- Rust (+7): QA jsonl/missing ai/diffusion guard, classification auto-tags, corpus sort, image negative_prompt, ChatXML cot=false

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 141
pytest: 322 passed
ruff: clean
```

## Merge-ready

YES
