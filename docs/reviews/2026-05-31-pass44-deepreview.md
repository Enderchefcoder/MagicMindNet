# Deep review pass 44 — 2026-05-31

## Scope

Merge weight averaging proof, ffn2 import shape coverage, quantize head/int4 block parity, checkpoint-strict subagent.

## Changes

- Rust tests (+5): merge block attn.q, merge classifier head, ffn2 shape, int4 block ffn, int8 classifier head
- Python tests (+4): merge chatbot/classifier export JSON averaging, ffn2 shape, int8 head quantize
- Docs: `checkpoints.md` merge averaging note; `testing.md` import/merge test refs
- Subagent: `.cursor/agents/magicmindnet-checkpoint-strict.md`

## TDD

- Regression tests only (existing merge/quantize/import behavior); no production code changes

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 91
pytest: 201 passed
ruff: clean
```

## Merge-ready

YES
