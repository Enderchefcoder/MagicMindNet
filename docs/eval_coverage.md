# Eval harness coverage matrix

Offline regression for `magicmindnet.eval` / `ai.run_suite` / `ai.EvalHarness`.

## Public API

| Symbol | Test |
|--------|------|
| `EvalHarness`, `BenchmarkRunner`, `Metric`, `TaskResult`, `SuiteReport` | `test_eval_public_exports` |
| `list_tasks`, `list_suites`, `get_task`, `run_suite` | `test_eval_exports_on_package_all` |
| `ai.__all__` exports | `test_eval_exports_on_package_all` |
| CLI `python -m magicmindnet.eval` / `examples/eval_harness.py` | `test_eval_harness_example_script` |

## Suites

| Suite | Guaranteed by |
|-------|---------------|
| `smoke`, `lm`, `cls`, `diffusion`, `io`, `hub`, `glint`, `generate`, `rl`, `train`, `all` | `test_list_suites_includes_canonical` |
| `smoke` end-to-end pass | `test_smoke_suite_runs_and_passes` |
| `all` ≥30 tasks, zero failures | `test_all_suite_completes` |

## Tasks (feature surface)

| Task | Suites | Notes |
|------|--------|-------|
| `lm_qa_loss` | smoke, lm | Mean CE |
| `lm_qa_train` | smoke, lm, train | Loss decreases |
| `lm_corpus_loss` / `lm_corpus_train` | lm / train | Corpus CE |
| `lm_learned_pe_train` | lm, train | Learned `pos_embed` |
| `lm_rope_train` | lm, train | RoPE |
| `lm_bpe_train` | lm, train | BPE encoder |
| `lm_unigram_train` | lm, train | Unigram encoder |
| `lm_gqa_train` | lm, train | `n_kv_heads < n_heads` |
| `lm_head_dim_train` | lm, train | Optional `head_dim` |
| `lm_glint_train` | smoke, lm, glint, train | loops/RMS/SwiGLU/tied |
| `lm_generate_latency` | smoke, lm, generate | `generate` wall time |
| `lm_vision_flag` | smoke, lm | `vision=True` construct + generate |
| `cls_mean_loss` / `cls_train` / `cls_accuracy` | cls (+smoke/train) | Fixture `labels_small.json` |
| `diffusion_denoise` / `diffusion_edit_denoise` | diffusion | Fixed-t mean denoise |
| `diffusion_train` / `diffusion_edit_train` | diffusion, train | `TrainDiffusion` |
| `io_roundtrip_*` | io (+smoke safetensors) | safetensors, HF, GGUF, Q8, npz, pt |
| `arrays_npz_roundtrip` | smoke, io | `save_arrays` / `load_arrays` |
| `hub_*` | hub (+smoke) | Synthetic families offline |
| `rl_policy_smoke` / `spin_smoke` | rl, train | One-step smoke |
| `merge_chatbot_roundtrip` | smoke, lm, io | `ai.merge` |
| `quantize_int8_roundtrip` | lm, io | In-place int8 + reload |

Required task set asserted in `test_list_tasks_covers_feature_surface`.

## CI / smoke

| Hook | Command |
|------|---------|
| Example smoke | `examples/eval_harness.py smoke` in `scripts/smoke_examples.*` |
| Pytest | `tests/test_eval_harness_py.py` |
| Module CLI | `python -m magicmindnet.eval smoke` |

See also [benchmarks.md](benchmarks.md), [hub_coverage.md](hub_coverage.md), [training_coverage.md](training_coverage.md).
