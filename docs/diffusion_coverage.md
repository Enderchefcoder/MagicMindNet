# Diffusion coverage

Foundation VAE + UNet API (not production Stable Diffusion).

## API

| Behavior | Test |
|----------|------|
| `Diffusion()` constructs | `test_datasets.py`, `test_diffusion_repr.py` |
| `latent_channels` getter (default 4) | `test_diffusion_repr.py` |
| `smoke_step()` → finite UNet output | `test_diffusion_smoke_py.py`, `examples/diffusion_smoke.py` |
| Rust `training_step` finite | `diffusion_tests::training_step_output_finite` |
| `train_step_denoise` updates UNet | `diffusion_tests::train_step_denoise_updates_unet_and_reduces_loss` |
| `TrainDiffusion` on `DatasetImageGen` | `test_train_diffusion.py`, `train_diffusion_fixture_reduces_denoise_loss` |
| `denoise_loss_on_image(path, t)` | `test_train_diffusion.py` |
| Image manifest path resolve | `image_gen_resolve_image_path_relative_to_manifest` |
| NCHW tensor from disk image | `train_diffusion_fixture_reduces_denoise_loss` |

## Training

```python
import magicmindnet as ai

ds = ai.DatasetImageGen("manifest.json")
d = ai.Diffusion()
cfg = ai.TrainConfig(epochs=5, learning_rate=0.05, cuda=False)
ai.TrainDiffusion(d, ds, cfg)
```

See `examples/diffusion_train.py`.

## Limitations

See [limitations.md](limitations.md#diffusion): VAE is encode-only in training; no full sampler schedule yet.

## Running

```powershell
python examples/diffusion_smoke.py
python examples/diffusion_train.py
pytest tests/test_diffusion_smoke_py.py tests/test_train_diffusion.py -q
cargo test -p mmn-models diffusion_tests
cargo test -p mmn-train train_diffusion_fixture
```
