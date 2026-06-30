# mmn-py PyO3 bindings coverage

Post–pass 78, `crates/mmn-py` is split by domain. Python sees a single `magicmindnet._native` module; layout is for maintainers and agents.

## Module map

| Rust path | Python types / functions | Primary pytest |
|-----------|-------------------------|----------------|
| `datasets/qa.rs` | `DatasetQA` | `test_datasets.py`, `test_dataset_matrix_py.py` |
| `datasets/corpus.rs` | `DatasetCorpus` | `test_train_corpus_py.py`, `test_dataset_corpus_*.py` |
| `datasets/classification.rs` | `DatasetClassification` | `test_classifier_*.py`, `test_dataset_matrix_py.py` |
| `datasets/image.rs` | `DatasetImageGen`, `DatasetImageEdit` | `test_image_fixtures_py.py` |
| `models/chatbot.rs` | `Chatbot` | `test_train_*.py`, `test_io_checkpoint_matrix_py.py` |
| `models/classifier.rs` | `Classifier` | `test_classifier_*.py`, `test_io_classifier_matrix_py.py` |
| `models/diffusion.rs` | `Diffusion` | `test_diffusion_smoke_py.py` |
| `train_config.rs` | `TrainConfig` | `test_train_config_*.py`, `test_api_surface_py.py` |
| `train/mod.rs` | `Train`, `TrainClassifier`, `RL`, `SPIN` | `test_train_rl_spin_py.py`, `test_data_mismatch*.py` |
| `io/mod.rs` | `export`, `import_model`, `merge`, `quantize`, classifier IO | `test_io_*`, `test_import_*`, `test_merge_*` |
| `resource.rs` | `limit`, `limit_percent` | `test_limit.py`, `test_limit_no_suffix.py` |
| `errors.rs` | `CPUError`, `CUDAError`, `DataMismatchError`, … | `test_public_exceptions.py` |
| `lib.rs` | `#[pymodule]` registry only (~58 lines) | `test_mmn_py_bindings_py.py`, `test_api_surface_py.py` |

## Split completion

See [mmn_py_split_plan.md](mmn_py_split_plan.md) — **complete** as of pass 78. No Python submodule exposure; `python/magicmindnet/__init__.py` remains the public surface.

## When changing bindings

1. Edit the domain file under `crates/mmn-py/src/` (not only `lib.rs`).
2. Register new symbols in `lib.rs` `#[pymodule]`.
3. Export from `python/magicmindnet/__init__.py` and `docs/API.md` if public.
4. Run `.\scripts\verify_gate.ps1`.
5. Extend this table and the relevant `*_coverage.md` doc.

## Regression smoke

`tests/test_mmn_py_bindings_py.py` — construct types from each major slice and roundtrip one chatbot checkpoint through `io/mod.rs` wrappers.
