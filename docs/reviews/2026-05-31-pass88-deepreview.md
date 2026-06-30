# Deep review pass 88 — 2026-05-31

## Scope

RL/SPIN policy for learned position embeddings; parameter count; import shape strictness.

## Changes

- **RL frozen** — `test_rl_does_not_change_learned_pos_embed` (RL only touches `lm_head`)
- **SPIN trains PE** — `test_spin_changes_learned_pos_embed` (Train phase in SPIN loop)
- **Parameter count** — `learned_pos_embed_increases_parameter_count` (Rust), `test_learned_pos_embed_increases_parameters` (Python)
- **Import strict** — `import_rejects_pos_embed_shape_mismatch` when meta `max_seq_len` disagrees with tensor shape

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 195 (+2)
pytest: 440 (+3)
ruff: clean
```

## Merge-ready

YES

## Next (pass 89)

- `export_bin` / `import_bin` roundtrip for learned `pos_embed`
- `docs/API.md` position-encoding constructor kwargs
