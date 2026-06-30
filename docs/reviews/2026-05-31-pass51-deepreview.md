# Deep review pass 51 — 2026-05-31

## Scope

Classifier IO matrix, multi-block chatbot IO (`n_layer=2`), quickstart venv fix.

## Changes

- `tests/test_io_classifier_matrix_py.py` — backbone/head missing, shape, merge, quantize (int8/int4)
- `tests/test_io_multiblock_chatbot_py.py` — all 10 `blocks.1.*` missing keys, merge block1, roundtrip
- Rust (+3): `import_rejects_missing_block1_ffn/attn_q`, `merge_models_averages_block1_ffn`
- `tests/test_quickstart.py` — prefers `MagicMindNet/.venv` Python
- `docs/checkpoint_coverage.md` — classifier matrix + multi-block sections

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 132
pytest: 310 passed
ruff: clean
```

## Merge-ready

YES
