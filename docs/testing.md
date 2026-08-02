# Testing

## Commands

```powershell
cargo test --workspace
maturin develop --release -m crates/mmn-py/Cargo.toml
pytest -q
.\scripts\lint.ps1
.\scripts\ci_local.ps1
.\scripts\count_tests.ps1
.\scripts\verify_gate.ps1
```

Linux/macOS merge gate (when `ci_local.sh` exists):

```bash
bash scripts/verify_gate.sh
bash scripts/ci_local.sh
bash scripts/count_tests.sh
bash scripts/lint.sh
```

Install dev tools: `pip install -e ".[dev]"` (pytest, maturin, ruff).

Gate scripts (`ci_local`, `smoke_examples`, `count_tests`, `verify_gate`) use `scripts/venv_python.*` to pick `.venv` Python when present — shell activation is optional on Windows.

## Coverage documentation index

Regression matrices live under `docs/*_coverage.md`. Extend the relevant doc when adding behavior.

| Doc | Scope |
|-----|--------|
| [checkpoint_coverage.md](checkpoint_coverage.md) | Chatbot/classifier IO contract (100% tensor keys) |
| [interop.md](interop.md) | Global formats: GGUF, PyTorch `.pt`, NumPy `.npy`/`.npz`, universal `ai.load` |
| [hub.md](hub.md) | `ai.from_pretrained` — HF / ModelScope / Ollama / local |
| [hub_coverage.md](hub_coverage.md) | Hub offline + hands-on coverage matrix |
| [distribai.md](distribai.md) | DistribAI grid bridge: architecture configs, script jobs, checkpoints (coverage matrix inside) |
| [benchmarks.md](benchmarks.md) | Unified eval harness + legacy example benches |
| [eval_coverage.md](eval_coverage.md) | `magicmindnet.eval` task/suite matrix |
| [training_coverage.md](training_coverage.md) | `Train`, `TrainClassifier`, RL/SPIN, mean loss |
| [dataset_coverage.md](dataset_coverage.md) | QA, corpus, classification, image loaders |
| [examples_coverage.md](examples_coverage.md) | Runnable `examples/` × smoke × pytest |
| [optimizers_coverage.md](optimizers_coverage.md) | AdamW, Muon, hybrid |
| [attention_coverage.md](attention_coverage.md) | Scaled dot-product attn; Train updates attn; RL frozen |
| [position_encoding_coverage.md](position_encoding_coverage.md) | Sinusoidal PE on embed; RoPE roadmap |
| [layernorm_coverage.md](layernorm_coverage.md) | LN forward/backward; γ/β train in LM |
| [nn_coverage.md](nn_coverage.md) | `mmn-nn` unit tests (block, attn, GELU) |
| [classifier_coverage.md](classifier_coverage.md) | Classifier train/predict edge cases |
| [vision_coverage.md](vision_coverage.md) | Vision-flag metadata path |
| [quantize_coverage.md](quantize_coverage.md) | int8/int4 quantize parity |
| [image_coverage.md](image_coverage.md) | ImageGen/ImageEdit fixtures |
| [diffusion_coverage.md](diffusion_coverage.md) | Diffusion smoke / dataset validation |
| [feature_parity.md](feature_parity.md) | PyTorch / llama.cpp / Ollama parity matrix (Wave 1+) |
| [limitations.md](limitations.md) | Alpha gaps and post-alpha roadmap |
| [mmn_py_coverage.md](mmn_py_coverage.md) | Split `mmn-py` module map + binding smoke tests |
| [mmn_py_split_plan.md](mmn_py_split_plan.md) | `mmn-py` split (complete — pass 78) |

See also [CONTRIBUTING.md](../CONTRIBUTING.md) (coverage table for contributors).

## Coverage areas (quick map)

| Area | Rust | Python |
|------|------|--------|
| Tensor / CE / embed backward | `mmn-core` | — |
| Optimizers | `mmn-optim` | `test_optimizer_integration_py.py` |
| Beginner API (`model.train/save/load/chat`, `ai.load`) | `mmn-io detect`, `mmn-train` | `test_model_train_methods.py`, `test_model_save_load.py`, `test_chatbot_chat.py`, `test_constructor_validation.py` |
| In-memory datasets (`data=[...]`) | — | `test_dataset_in_memory.py` |
| Typing (`py.typed`, `_native.pyi`) | — | `test_type_stubs.py` |
| Datasets / ChatXML | `mmn-data` | `test_dataset_matrix_py.py`, `test_datasets.py`, … |
| `mmn-nn` blocks | `mmn-nn` | — (see [nn_coverage.md](nn_coverage.md)) |
| Train updates attn + LN (RL frozen) | `mmn-models` | `test_train_block_params_py.py`, `test_train_rl_spin_py.py` |
| Classifier CUDA gate | `mmn-train` | `test_train_classifier_cuda.py` |
| Chatbot train / RL | `mmn-train`, `mmn-models` | `test_train*.py`, `test_mean_qa_loss.py`, … |
| Classifier train / IO | `mmn-io`, `mmn-models` | `test_classifier_*.py`, `test_io_classifier_matrix_py.py`, … |
| PyO3 API / errors | — | `test_public_exceptions.py`, `test_api_surface_py.py`, `test_mmn_py_bindings_py.py` |
| Checkpoints / merge | `mmn-io` | `test_io_checkpoint_matrix_py.py`, `test_import_*`, `test_merge_*`, … |
| Global formats (GGUF / `.pt` / `.npy`/`.npz`) | `mmn-io interop` | `test_interop_*.py`, `test_universal_formats_py.py` |
| Checkpoint train deltas | — | `conftest.checkpoint_tensor_*`, `tensor_entry_first_f32` |
| Conftest / CI python | — | `test_conftest_helpers_py.py` |
| Quantize | `mmn-io` | `test_io_ln_quantize_py.py`, `test_quantize_*.py`, … |
| Example scripts | — | `test_examples_scripts_py.py` (via `conftest.run_example`) |

Re-run the commands above after changes; counts drift as tests are added (`.\scripts\count_tests.ps1`).
