# Deep review pass 57 — 2026-05-31

## Scope

Corpus LM training, diffusion smoke, mmn-io module extraction.

## Changes

- **`train_corpus` / `mean_corpus_loss`** — `Train(chatbot, DatasetCorpus)` next-token LM; `compute_mean_loss` accepts corpus
- **`Diffusion.smoke_step()`** — Python + Rust finite training-step check
- **`mmn-io/src/checkpoint_util.rs`** — extracted tensor JSON helpers (+ roundtrip test)
- Examples: `corpus_benchmark.py`, `diffusion_smoke.py`; fixtures `corpus_rows.json`, `corpus.txt`
- Docs: `diffusion_coverage.md`; API/training/dataset coverage updates

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 155
pytest: 391 passed
ruff: clean
```

## Merge-ready

YES
