# Diffusion coverage

Foundation VAE + UNet API (not production Stable Diffusion).

## API

| Behavior | Test |
|----------|------|
| `Diffusion()` constructs | `test_datasets.py`, `test_diffusion_repr.py` |
| `latent_channels` getter (default 4) | `test_diffusion_repr.py` |
| `smoke_step()` → finite UNet output | `test_diffusion_smoke_py.py`, `examples/diffusion_smoke.py` |
| Rust `training_step` finite | `diffusion_tests::training_step_output_finite` |

## Limitations

See [limitations.md](limitations.md#diffusion): conv paths are structural stubs; no full image dataset training loop yet.

## Running

```powershell
python examples/diffusion_smoke.py
pytest tests/test_diffusion_smoke_py.py -q
cargo test -p mmn-models training_step_output_finite
```
