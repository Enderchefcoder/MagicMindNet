# Examples

Runnable demos live here. From the repo root, activate `.venv` and run `maturin develop --release` first.

| Script | Purpose |
|--------|---------|
| [quickstart.py](quickstart.py) | Minimal QA load → train → export (optional `--learned-pe`, `--rope`, `--bpe`) |
| [benchmark_train.py](benchmark_train.py) | Mean QA loss before/after `Train` (optional `--learned-pe`, `--rope`, `--bpe`) |
| [bpe_roundtrip.py](bpe_roundtrip.py) | BPE `save`/`load` parity + optional `--train` with `bpe_encoder` |
| [rl_spin.py](rl_spin.py) | `RL` then `SPIN` on fixture QA |
| [checkpoint_roundtrip.py](checkpoint_roundtrip.py) | Chatbot export/import parity |
| [learned_pos_embed_roundtrip.py](learned_pos_embed_roundtrip.py) | Learned `pos_embed` export/import + mean-loss parity (optional `--train`) |
| [rope_roundtrip.py](rope_roundtrip.py) | RoPE chatbot export/import + mean-loss parity (optional `--train`) |
| [classifier_roundtrip.py](classifier_roundtrip.py) | Classifier export/import parity |
| [classification.py](classification.py) | Train classifier, export, predict |
| [classification_benchmark.py](classification_benchmark.py) | Classification mean loss before/after train |
| [corpus_benchmark.py](corpus_benchmark.py) | Corpus LM mean loss before/after `Train` (optional `--learned-pe`, `--rope`, `--bpe`) |
| [diffusion_smoke.py](diffusion_smoke.py) | `Diffusion.smoke_step()` finite output check |
| [diffusion_train.py](diffusion_train.py) | Single-image denoise loss before/after `TrainDiffusion` |
| [diffusion_benchmark.py](diffusion_benchmark.py) | Mean denoise loss delta on `image_gen` or `--edit` inpainting |
| [diffusion_sample.py](diffusion_sample.py) | `sample_rgb_patch` demo |
| [diffusion_roundtrip.py](diffusion_roundtrip.py) | Train, export/import, sample parity |
| [diffusion_edit_train.py](diffusion_edit_train.py) | Inpaint training smoke |
| [diffusion_inpaint_sample.py](diffusion_inpaint_sample.py) | Inpaint sample + PNG export |
| [diffusion_edit_roundtrip.py](diffusion_edit_roundtrip.py) | Inpaint train + checkpoint roundtrip |
| [diffusion_quantize_roundtrip.py](diffusion_quantize_roundtrip.py) | int8 quantize + export/import sample parity |
| [vision_chatbot.py](vision_chatbot.py) | Vision-flag chatbot train + export roundtrip |
| [eval_mean_loss.py](eval_mean_loss.py) | `qa`, `cls`, `corpus`, `diffusion`, or `diffusion-edit` (add `--train`, `--learned-pe`, `--rope`, `--bpe`) |

## Quick commands

```powershell
python examples/quickstart.py
python examples/benchmark_train.py
python examples/rl_spin.py
python examples/eval_mean_loss.py qa
python examples/classification.py
python examples/classification_benchmark.py
```

Full gate (includes quickstart + roundtrips + eval):

```powershell
.\scripts\smoke_examples.ps1
```

See also [docs/API.md](../docs/API.md), [docs/training.md](../docs/training.md), and [docs/checkpoints.md](../docs/checkpoints.md).
