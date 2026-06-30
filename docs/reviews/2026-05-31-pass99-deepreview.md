# Deep review pass 99 — 2026-05-31

## Scope

`eval_mean_loss` learned PE flag for chatbot modes.

## Changes

- **Example** — `eval_mean_loss.py qa|corpus [--learned-pe]` with `_make_chatbot` helper
- **pytest** — `test_eval_mean_loss_qa_learned_pe_runs`, `test_eval_mean_loss_corpus_learned_pe_runs`
- **Docs** — `API.md`, `examples/README.md`, `position_encoding_coverage.md`

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 202 (unchanged)
pytest: 460 (+2)
```

## Merge-ready

YES

## Next (pass 100)

- `eval_mean_loss qa --train --learned-pe` pytest
- `training_coverage.md` learned PE example flags row
