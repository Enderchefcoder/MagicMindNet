# Vision chatbot path

The `Chatbot(vision=True)` flag enables a **conv + linear patch prefix encoder** that prepends a projected image row before text token embeddings in `forward_hidden` / training.

## What works (alpha)

| Behavior | Rust | Python |
|----------|------|--------|
| `has_vision` getter | `Chatbot::has_vision` | `bot.has_vision` |
| Patch encoder present | `has_vision_patch_encoder` | `bot.has_vision_patch_encoder` |
| RGB conv encoder | `has_vision_rgb_conv` | `bot.has_vision_rgb_conv` |
| Grayscale patch size (64 floats) | `VISION_PATCH_DIM` | `ai.VISION_PATCH_DIM`, `bot.vision_patch_dim` |
| RGB patch size (192 floats, NCHW) | `VISION_RGB_DIM` | `ai.VISION_RGB_DIM`, `bot.vision_rgb_dim` |
| Forward with patch prefix | `forward_hidden_with_patches` | `compute_loss(..., image_patch=...)` |
| Demo grayscale patch | `vision_patch_from_text` | `ai.vision_patch_from_text` |
| Demo RGB patch | `vision_rgb_patch_from_text` | `ai.vision_rgb_patch_from_text` |
| QA train uses input patch | `train` (RGB when conv loaded) | `ai.Train` on vision bot |
| Safetensors `vision_patch_proj` + `vision_patch_conv` | export/import | `test_vision_patch_encoder_py.py` |
| `bin` meta `vision_patch_dim` / `vision_rgb_dim` | `export_bin` | `safetensors_vision_patch_proj_roundtrip` (Rust) |
| Merge OR semantics + proj/conv average | `merge_models` | `test_merge_vision_or.py` |
| Quantize preserves flag + proj + conv | `quantize_model` | `test_vision_chatbot_py.py` |

## Patch pipeline

1. **RGB path (default for new vision models):** flat 192-float `NCHW` patch → `Conv2d(3→1, k=3)` → flatten 64 → `Linear(64→d_model)` → prepend row.
2. **Grayscale path (legacy checkpoints without `vision_patch_conv`):** flat 64-float patch → `Linear(64→d_model)`.

## Example

```powershell
python examples/vision_chatbot.py
```

```python
import magicmindnet as ai

rgb = ai.vision_rgb_patch_from_text("photo of a cat")
bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, vision=True)
loss = bot.compute_loss("describe this", "a cat", image_patch=rgb)
```

## Checkpoint keys

When `vision=true`, safetensors checkpoints include:

- **meta** `vision_patch_dim` (64), `vision_rgb_dim` (192), `vision_rgb_patch` (when conv present)
- **tensor** `vision_patch_proj` with shape `[d_model, 64]`
- **tensor** `vision_patch_conv` with shape `[1, 3, 3, 3]` (when RGB conv enabled)

## Merge

`merge(a, b)` sets `vision = a.vision || b.vision` and element-wise-averages `vision_patch_proj` and `vision_patch_conv` when present on either side. See [checkpoints.md](checkpoints.md).

## Limitations

- 8×8 surrogate patches from UTF-8 bytes (not disk image files yet).
- Corpus LM training does not attach patches (text-only); QA training uses `vision_rgb_patch_from_text(input)` when conv is loaded.
- No cross-attention between image and text streams yet.
- `DatasetImageGen` / `DatasetImageEdit` remain diffusion stubs, not `Chatbot` training.

## Diffusion / Conv2d

- `Conv2d::forward` applies real NCHW convolution with same padding (`kernel/2`) for VAE/UNet blocks.
- `vae_encoder_preserves_8x8_latent_shape` (Rust) and `Diffusion.smoke_step()` (Python) assert finite 8×8 latents.

## Roadmap

1. **Cross-attn block** — Q from text, K/V from image patches after self-attn.
2. **DatasetQA image paths** — load files from disk into normalized RGB patches.

See also [training.md](training.md), [checkpoint_coverage.md](checkpoint_coverage.md), and [limitations.md](limitations.md).
