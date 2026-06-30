# Deep review pass 86 — 2026-05-31

## Scope

Learned position embedding table (opt-in), checkpoint IO, bindings smoke, RoPE doc expansion.

## Changes

- **`Chatbot::new_with_pe_options`** — `use_learned_pos_embed`, `max_seq_len` (default 512); sinusoidal remains default
- **Train/backward** — `apply_position_encoding`, `pos_embed` grad + optim step when enabled
- **IO** — meta `use_learned_pos_embed` / `max_seq_len`; tensor `pos_embed`; merge averages; quantize includes PE
- **Python** — `Chatbot(..., use_learned_pos_embed=, max_seq_len=)`; getters `use_learned_pos_embed`, `max_seq_len`
- **Tests** — `learned_pos_embed_roundtrip_preserves_weights`, `import_rejects_missing_pos_embed_when_meta_requires`, `test_learned_pos_embed_io_py.py`, bindings smoke
- **`docs/position_encoding_coverage.md`** — learned PE table + RoPE design sketch

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 189 (+3)
pytest: 434 (+2)
ruff: clean
```

## Merge-ready

YES

## Next (pass 87)

- merge / quantize regression tests for learned `pos_embed`
- `Train()` pytest asserting `pos_embed` weight change
- merge rejects mismatched PE settings
