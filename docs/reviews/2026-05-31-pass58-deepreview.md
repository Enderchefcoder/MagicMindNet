# Deep review pass 58 — 2026-05-31

## Scope

Image dataset fixtures, RL mode improvements, mmn-io block_tensors split.

## Changes

- **`mmn-io/src/block_tensors.rs`** — extracted chatbot block export/import from `lib.rs`
- **RL modes** — `reward_only`/`selfplay` skip punished rows; new `punish_only` mode; +2 Rust tests
- **Image datasets** — `image_edit_loads_mask_and_negative_prompt`; fixtures `image_gen.json`, `image_edit.json`; `test_image_fixtures_py.py`
- **Diffusion validation** — corpus rejected, image_gen accepted (Rust tests)
- **Docs** — `image_coverage.md`; training/dataset coverage RL + image updates

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 160
pytest: 395 passed
ruff: clean
```

## Merge-ready

YES
