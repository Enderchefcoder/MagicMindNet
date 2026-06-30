# Deep review pass 89 — 2026-05-31

## Scope

`bin` format learned PE meta roundtrip; API docs; `max_seq_len` guard tests.

## Changes

- **`export_bin` / `import_bin`** — persist `use_learned_pos_embed` + `max_seq_len`; import via `new_with_pe_options`
- **Tests** — `bin_learned_pos_embed_roundtrip_preserves_meta`, `test_bin_roundtrip_preserves_learned_pos_embed_meta`, `learned_pos_embed_rejects_long_sequence`, `test_compute_loss_rejects_sequence_longer_than_max_seq_len`
- **Docs** — `API.md` position-encoding table + getters; `checkpoints.md` bin/safetensors PE keys

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 197 (+2)
pytest: 442 (+2)
ruff: clean
```

## Merge-ready

YES

## Next (pass 90)

- `Train()` with learned PE reduces mean loss (pytest)
- `limitations.md` bin PE meta note
