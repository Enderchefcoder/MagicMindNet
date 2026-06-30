# Deep review pass 41 — 2026-05-31

## Scope

Complete safetensors meta strictness (vocab_size), corrupt file paths, classifier head validation, bin defaults docs.

## Changes

- **Fix:** `import_safetensors` requires `meta.vocab_size` (removed caller-arg fallback)
- Rust tests (+6): missing vocab meta, invalid JSON, empty file, missing head, head shape mismatch, bin `{}` defaults
- Python tests (+6): mirror import errors, bin defaults, corpus `row` batch
- Docs: `checkpoints.md` vocab required + bin empty defaults

## TDD

- `import_rejects_missing_vocab_size_meta` written with fix (1 red→green on meta strictness)

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 73
pytest: 177 passed
ruff: clean
```

## Merge-ready

YES
