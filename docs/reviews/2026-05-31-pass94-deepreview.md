# Deep review pass 94 — 2026-05-31

## Scope

Parametric IO checkpoint matrix coverage for learned `pos_embed`.

## Changes

- **`test_io_checkpoint_matrix_py.py`** — learned PE helpers + 6 matrix cases (missing, shape, merge, int8/int4 quantize)
- **`test_export_import_preserves_learned_pos_embed_compute_loss`** — mean loss parity after roundtrip
- **`checkpoint_coverage.md`** / **`position_encoding_coverage.md`** — matrix references updated

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 201 (unchanged)
pytest: 456 (+6)
ruff: clean
```

## Merge-ready

YES

## Next (pass 95)

- Rust `import_preserves_forward_loss` for learned `pos_embed`
- Optional quickstart / examples note for learned PE flag
