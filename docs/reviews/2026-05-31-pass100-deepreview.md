# Deep review pass 100 — 2026-05-31

## Scope

Train + learned PE example pytest; training coverage matrix.

## Changes

- **pytest** — `test_eval_mean_loss_qa_learned_pe_train_runs`, `test_eval_mean_loss_corpus_learned_pe_train_runs`
- **Docs** — `training_coverage.md` learned-PE example flags tables (fixed orphaned rows)

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
pytest: 462 (+2)
```

## Merge-ready

YES
