# Deep review pass 33 — 2026-05-31

## Scope

Public IO aliases, classifier constructor labels, Chatbot `tokenizer` getter, Diffusion `latent_channels`, Linux `verify_gate.sh`, docs.

## Changes

- `tests/test_public_io_surface.py` — callable `export` / `import_model` / classifier aliases / merge / limit
- `tests/test_classifier_num_labels_constructor.py` — `Classifier(4, …)` → `num_labels` and default label names
- `tests/test_chatbot_tokenizer_getter.py` — non-empty `tokenizer` string
- `tests/test_diffusion_latent_channels.py` — `latent_channels >= 1`
- `scripts/verify_gate.sh` — bash merge gate mirroring `verify_gate.ps1`
- `docs/testing.md`, `CONTRIBUTING.md` — document `verify_gate.sh` and new tests

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 50
pytest: 120 passed
ruff: clean
```

## Merge-ready

YES (in-repo gate green).
