# Deep review pass 96 — 2026-05-31

## Scope

Pytest smoke for learned PE roundtrip example; API docs link.

## Changes

- **pytest** — `test_learned_pos_embed_roundtrip_example_runs` asserts stdout markers
- **Docs** — `docs/API.md` constructor note + examples table; `examples_coverage.md` pytest=yes

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 202 (unchanged)
pytest: 457 (+1)
examples smoke: learned pos_embed roundtrip OK
```

## Merge-ready

YES

## Next (pass 97)

- README examples table row for learned PE roundtrip
- `position_encoding_coverage.md` pytest smoke row
