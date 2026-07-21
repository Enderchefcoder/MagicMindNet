# Examples coverage matrix

Runnable scripts under `examples/` and how they are regression-tested.

| Script | Purpose | Smoke (`smoke_examples`) | pytest (`test_examples_scripts_py`) |
|--------|---------|--------------------------|-------------------------------------|
| [pip_quickstart.py](../examples/pip_quickstart.py) | Post-`pip install` train → chat → save → load | yes | yes |
| [train_glint2.py](../examples/train_glint2.py) | **Glint-2 exact** FineWeb-Edu trainer (MMN-only; coda/window/LoopLoRA) | demo | yes |
| [generate_glint2.py](../examples/generate_glint2.py) | Glint-2 sampling defaults + official tokenizer | — | via train_glint2 |
| [hello_ai.py](../examples/hello_ai.py) | Beginner tour: in-memory data, `bot.train`/`chat`/`save`, `ai.load`, `predict_label` | yes | yes |
| [openai_server_demo.py](../examples/openai_server_demo.py) | Local OpenAI `/v1/chat/completions` smoke | yes | yes |
| [hub_catalog.py](../examples/hub_catalog.py) | Hub family catalog + local `from_pretrained` | yes | yes |
| [parity_demo.py](../examples/parity_demo.py) | Stream / embed / typical_p / chat_messages / tools | yes | yes |
| [quickstart.py](../examples/quickstart.py) | Minimal QA train + export; `--learned-pe` | yes | yes |
| [benchmark_train.py](../examples/benchmark_train.py) | QA mean loss before/after `Train`; `--learned-pe` | yes | yes |
| [eval_mean_loss.py](../examples/eval_mean_loss.py) | Mean CE (`qa` / `cls` / `corpus`) or denoise loss (`diffusion` / `diffusion-edit`); `--learned-pe`, `--train` | yes (all modes) | yes (+ flag variants) |
| [corpus_benchmark.py](../examples/corpus_benchmark.py) | Corpus LM train delta; `--learned-pe` for learned `pos_embed` | yes | yes |
| [classification.py](../examples/classification.py) | Classifier train + predict | yes | yes |
| [classification_benchmark.py](../examples/classification_benchmark.py) | Classification train delta | yes | yes |
| [checkpoint_roundtrip.py](../examples/checkpoint_roundtrip.py) | Chatbot export/import | yes | yes |
| [global_formats_roundtrip.py](../examples/global_formats_roundtrip.py) | GGUF / `.pt` / `.npz` roundtrips + generic array IO | yes | yes |
| [interop_benchmark.py](../examples/interop_benchmark.py) | Format save/load timing + sizes (incl. k-quants) | yes | yes |
| [learned_pos_embed_roundtrip.py](../examples/learned_pos_embed_roundtrip.py) | Learned `pos_embed` export/import + loss; `--train` | yes | yes |
| [rope_roundtrip.py](../examples/rope_roundtrip.py) | RoPE chatbot export/import + loss; `--train` | yes | yes |
| [classifier_roundtrip.py](../examples/classifier_roundtrip.py) | Classifier export/import | yes | yes |
| [rl_spin.py](../examples/rl_spin.py) | `RL` + `SPIN` on fixture QA | yes | yes |
| [diffusion_smoke.py](../examples/diffusion_smoke.py) | `Diffusion.smoke_step()` | yes | yes |
| [diffusion_benchmark.py](../examples/diffusion_benchmark.py) | Mean denoise loss before/after `TrainDiffusion`; `--edit` for inpainting | yes | yes |
| [diffusion_roundtrip.py](../examples/diffusion_roundtrip.py) | Train, export/import, sample parity | yes | yes |
| [diffusion_edit_roundtrip.py](../examples/diffusion_edit_roundtrip.py) | Inpaint train + export/import inpaint sample parity | yes | yes |
| [diffusion_quantize_roundtrip.py](../examples/diffusion_quantize_roundtrip.py) | int8 quantize + export/import sample parity | yes | yes |
| [diffusion_merge_demo.py](../examples/diffusion_merge_demo.py) | `merge_diffusion` sample smoke | yes | yes |
| [vision_chatbot.py](../examples/vision_chatbot.py) | Vision-flag train + export | yes | yes |
| [hub_local_roundtrip.py](../examples/hub_local_roundtrip.py) | `ai.from_pretrained` local Chatbot generate/finetune/save | yes | via hub wave2 |
| [glint_tiny.py](../examples/glint_tiny.py) | Glint-style RMSNorm/SwiGLU/`n_loops`/tie train smoke | yes | via glint tests |
| [eval_harness.py](../examples/eval_harness.py) | Unified `magicmindnet.eval` suites + JSON | yes (`smoke`) | `test_eval_harness_py.py` |
| [hub_hands_on.py](../scripts/hub_hands_on.py) (`--only local`) | Offline hub matrix smoke | yes | yes (`test_hands_on_local_subprocess`) |

## Running

```powershell
.\scripts\smoke_examples.ps1
pytest tests/test_examples_scripts_py.py -q
```

Shared harness: `tests/conftest.py` (`run_example` fixture with optional CLI args).

See also [examples/README.md](../examples/README.md) and [training_coverage.md](training_coverage.md).
