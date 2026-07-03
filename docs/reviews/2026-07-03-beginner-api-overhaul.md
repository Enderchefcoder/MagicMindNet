# Deep review — beginner API overhaul — 2026-07-03

## Scope

Make MagicMindNet an easy-to-use, professional beginner library: fix silent/panicking
error paths, add method-style workflows, universal loading, in-memory datasets,
training feedback, IDE typing, and a beginner documentation track.

## Bugs fixed

- `optimizer="muon"` silently trained with plain AdamW (`use_hybrid` only checked
  `== "hybrid"`); now routes matrices through Muon and unknown names raise `ValueError`
- `Chatbot(use_learned_pos_embed=True, use_rope=True)` raised `pyo3_runtime.PanicException`;
  now `ValueError` with fix guidance
- Unknown `autoset` presets silently fell back to sub-100M; now `ValueError` listing presets
- `vocab_size=0` accepted at construction; now `ValueError`
- Reprs printed Rust-style `vision=false` / `cuda=false`; now Python `True`/`False`
- `python/magicmindnet/vision.py` had corrupted formatting and inline imports
- Crate-local `target/` dirs and the maturin-built `.so` were not gitignored

## Changes

- **mmn-train** — `TrainConfig.verbose`; `VALID_OPTIMIZERS` + `resolve_use_hybrid`;
  `train*`, `train_classifier`, `train_diffusion*` return `Vec<f32>` per-epoch mean losses
- **mmn-io** — new `detect.rs`: `CheckpointKind` + `detect_checkpoint_kind` for all
  JSON (`mmn-safetensors-v1`, `mmn-classifier-v1`, `mmn-diffusion-v1`, `mmn-bin-v1`)
  and binary HF safetensors (chatbot/classifier/external) files
- **mmn-py** — `Chatbot.save/load/train/chat`; `Classifier.save/load/train/predict_label`;
  `Diffusion.save/load/train`; universal `load` / `load_checkpoint`; ctor validation;
  `TrainConfig` verbose + optimizer validation; shared `resolve_train_config`;
  Python-stdout flush before Rust progress prints
- **Datasets** — in-memory `DatasetQA(data=[...])`, `DatasetClassification(data=[...])`,
  `DatasetCorpus(data=[...])` (`format == "memory"`); file/data exclusivity validation
- **Python pkg** — `py.typed` + complete `_native.pyi`; `load` exports; pyproject
  keywords/URLs/`Typing :: Typed`
- **Docs** — `getting_started.md` tutorial; beginner-first README; API.md;
  training/dataset/checkpoint/examples/testing coverage matrices; CHANGELOG
- **Examples** — `hello_ai.py` (+ smoke scripts + pytest)

## Verification

```
bash scripts/ci_local.sh  → CI local: OK
  (cargo test --workspace, maturin develop --release, pytest, ruff, quickstart, smoke examples)
Rust #[test]: 314 → 325 (+11)
pytest: 640 → 714 (+74)
```

## Merge-ready

YES

## Next

- Consider `RL`/`SPIN` method-style wrappers if beginner demand appears
- SentencePiece-scale unigram vocab (existing roadmap, `limitations.md`)
