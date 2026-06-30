# Examples coverage matrix

Runnable scripts under `examples/` and how they are regression-tested.

| Script | Purpose | Smoke (`smoke_examples`) | pytest (`test_examples_scripts_py`) |
|--------|---------|--------------------------|-------------------------------------|
| [quickstart.py](../examples/quickstart.py) | Minimal QA train + export; `--learned-pe` | yes | yes |
| [benchmark_train.py](../examples/benchmark_train.py) | QA mean loss before/after `Train`; `--learned-pe` | yes | yes |
| [eval_mean_loss.py](../examples/eval_mean_loss.py) | Mean CE (`qa` / `cls` / `corpus`) or denoise loss (`diffusion` / `diffusion-edit`); `--learned-pe`, `--train` | yes (all modes) | yes (+ flag variants) |
| [corpus_benchmark.py](../examples/corpus_benchmark.py) | Corpus LM train delta; `--learned-pe` for learned `pos_embed` | yes | yes |
| [classification.py](../examples/classification.py) | Classifier train + predict | yes | yes |
| [classification_benchmark.py](../examples/classification_benchmark.py) | Classification train delta | yes | yes |
| [checkpoint_roundtrip.py](../examples/checkpoint_roundtrip.py) | Chatbot export/import | yes | yes |
| [learned_pos_embed_roundtrip.py](../examples/learned_pos_embed_roundtrip.py) | Learned `pos_embed` export/import + loss; `--train` | yes | yes |
| [rope_roundtrip.py](../examples/rope_roundtrip.py) | RoPE chatbot export/import + loss; `--train` | yes | yes |
| [classifier_roundtrip.py](../examples/classifier_roundtrip.py) | Classifier export/import | yes | yes |
| [rl_spin.py](../examples/rl_spin.py) | `RL` + `SPIN` on fixture QA | yes | yes |
| [diffusion_smoke.py](../examples/diffusion_smoke.py) | `Diffusion.smoke_step()` | yes | yes |
| [diffusion_benchmark.py](../examples/diffusion_benchmark.py) | Mean denoise loss before/after `TrainDiffusion`; `--edit` for inpainting | yes | yes |
| [vision_chatbot.py](../examples/vision_chatbot.py) | Vision-flag train + export | yes | yes |

## Running

```powershell
.\scripts\smoke_examples.ps1
pytest tests/test_examples_scripts_py.py -q
```

Shared harness: `tests/conftest.py` (`run_example` fixture with optional CLI args).

See also [examples/README.md](../examples/README.md) and [training_coverage.md](training_coverage.md).
