# mmn-py module split plan (complete)

`crates/mmn-py/src/lib.rs` was ~780 lines — split **without changing the Python API** (`import magicmindnet as ai`). Completed pass 78; see [mmn_py_coverage.md](mmn_py_coverage.md) for the module map.

## Goals

- Keep `lib.rs` as a thin `#[pymodule]` registry (like post-split `mmn-io`)
- Isolate PyO3 glue by domain for review and agent navigation
- No new cyclic deps; `mmn-py` continues to depend on `mmn-*` crates only

## Proposed layout

```
crates/mmn-py/src/
  lib.rs              # pymodule registry only (~58 lines) **[done pass 78]**
  errors.rs           # create_exception! + mmn_err_to_py (optional move from lib)
  train_config.rs     # PyTrainConfig
  datasets/
    mod.rs
    qa.rs             # PyDatasetQA
    corpus.rs         # PyDatasetCorpus
    classification.rs # PyDatasetClassification
    image.rs          # PyDatasetImageGen, PyDatasetImageEdit
  models/
    mod.rs
    chatbot.rs        # PyChatbot
    classifier.rs     # PyClassifier
    diffusion.rs      # PyDiffusion
  train/
    mod.rs            # Train, TrainClassifier, RL, SPIN
  io/
    mod.rs            # export/import/merge/quantize (chatbot + classifier)
  resource.rs         # limit_resources, limit_percent
```

## Migration order (TDD)

1. **errors.rs** — move exception macro + `mmn_err_to_py`; `cargo test -p mmn-py` unchanged (no py tests in crate; gate = maturin + pytest public API). **[done pass 71]**
2. **train_config.rs** — smallest pyclass. **[done pass 72]**
3. **resource.rs** — `limit` / `limit_percent`. **[done pass 72]**
4. **datasets/** — four dataset pyclasses (~350 lines). **[done pass 75]** (`qa`, `corpus`, `classification`, `image`)
5. **models/** — three model pyclasses (~220 lines). **[done pass 77]** (`chatbot`, `classifier`, `diffusion`)
6. **train/** + **io/** — free functions last (highest coupling to models). **[done pass 78]**
7. **lib.rs** — only `#[pymodule]` registration. **[done pass 78 — 58 lines]**

Each step: move code, `maturin develop`, `pytest tests/test_public_exceptions.py tests/test_api_surface_py.py -q`, full `verify_gate`.

## Out of scope (this pass)

- Renaming `Train` / `RL` to snake_case (public API freeze)
- Splitting `python/magicmindnet/__init__.py` (thin re-export today)
- New PyO3 submodules exposed to Python

## References

- Prior art: `mmn-io` split (`chatbot_io`, `classifier_io`, `io_tests/`) — passes 60–62
- Agent: `.cursor/agents/magicmindnet-python.md`
- Coverage unchanged: all bindings exercised via existing pytest matrix

## Acceptance

- `lib.rs` under ~150 lines — **58 lines (pass 78)**
- `verify_gate` green after each migration slice
- No change to `docs/API.md` symbol list
