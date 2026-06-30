# Deep review pass 98 — 2026-05-31

## Scope

Corpus benchmark with learned position embeddings.

## Changes

- **Example** — `corpus_benchmark.py --learned-pe` builds Chatbot with `use_learned_pos_embed=True`, `max_seq_len=128`
- **pytest** — `test_corpus_benchmark_learned_pe_example_runs`
- **Smoke** — `smoke_examples.ps1` runs learned-pe variant
- **Docs** — `examples/README.md`, `examples_coverage.md`, `position_encoding_coverage.md`

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 202 (unchanged)
pytest: 458 (+1)
corpus_benchmark --learned-pe: loss 5.55 → 4.42
```

## Merge-ready

YES

## Next (pass 99)

- `eval_mean_loss.py` optional `--learned-pe` for QA/corpus modes
- Or extend `training_coverage.md` corpus benchmark row
