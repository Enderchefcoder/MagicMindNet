# Deep review pass 95 — 2026-05-31

## Scope

Learned PE import loss parity (Rust); runnable example + docs.

## Changes

- **Rust** — `import_preserves_forward_loss_learned_pos_embed`
- **Example** — `examples/learned_pos_embed_roundtrip.py` wired into `smoke_examples.ps1`
- **Docs** — README learned PE snippet; quickstart comment; `examples_coverage.md`, `checkpoint_coverage.md`, `position_encoding_coverage.md`

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 202 (+1)
pytest: 456 (unchanged)
examples smoke: learned pos_embed roundtrip OK
```

## Merge-ready

YES

## Next (pass 96)

- `test_examples_scripts_py.py` smoke for `learned_pos_embed_roundtrip.py`
- `docs/API.md` link to learned PE example
