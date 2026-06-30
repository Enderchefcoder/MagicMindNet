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
| `TrainDiffusion` on `DatasetImageEdit` (masked inpainting) | `test_train_diffusion_edit.py`, `train_diffusion_edit_fixture_runs_masked_steps` |
| `sample_rgb_patch` values in `[0, 1]` after decode clamp | `test_train_diffusion_edit.py`, `sample_image_rgb_is_clamped_to_unit_interval` |
| `merge_diffusion` averages weights | `merge_diffusion_averages_unet_down_weight`, `test_train_diffusion_edit.py` |
| `quantize_diffusion` int8/int4 | `quantize_diffusion_int8_changes_unet_weight`, `test_diffusion_inpaint_py.py` |
| Inpaint sample preserves unmasked latent | `sample_latent_inpaint_preserves_unmasked_region` |
| `sample_inpaint_rgb_patch` / PNG export | `test_diffusion_inpaint_py.py`, `examples/diffusion_inpaint_sample.py` |
| `denoise_loss_on_image_masked` | `test_diffusion_inpaint_py.py` |
| `compute_mean_denoise_loss` on ImageGen/ImageEdit | `test_diffusion_mean_loss_py.py`, `mean_denoise_loss_*` |
| `eval_mean_loss.py diffusion` modes | `test_eval_mean_loss_diffusion_*` |
| Fixed-`t` training lowers mean denoise loss | `mean_denoise_loss_decreases_after_fixed_t_training` |
| Diffusion IO missing/shape/merge/quantize matrix | `test_io_diffusion_matrix_py.py` |
| Mask tensor from disk | `grayscale_mask_tensor_from_image_path_is_unit_interval` |
| `write_rgb_nchw_tensor_to_png` | `write_rgb_nchw_tensor_to_png_roundtrip` |
| `sample_rgb_patch(steps, seed)` finite + deterministic | `test_diffusion_sample_io_py.py`, `examples/diffusion_sample.py` |
| `export_diffusion` / `import_diffusion` roundtrip | `diffusion_export_import_roundtrip_preserves_sample`, `test_diffusion_sample_io_py.py` |
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
