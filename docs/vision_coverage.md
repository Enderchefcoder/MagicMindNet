# Vision chatbot path

The `Chatbot(vision=True)` flag enables a **linear patch prefix encoder** that prepends a projected 8×8 grayscale patch row before text token embeddings in `forward_hidden` / training.

## What works (alpha)

| Behavior | Rust | Python |
|----------|------|--------|
| `has_vision` getter | `Chatbot::has_vision` | `bot.has_vision` |
| Patch encoder present | `has_vision_patch_encoder` | `bot.has_vision_patch_encoder` |
| Patch size (64 floats) | `VISION_PATCH_DIM` | `ai.VISION_PATCH_DIM`, `bot.vision_patch_dim` |
| Forward with patch prefix | `forward_hidden_with_patches` | `compute_loss(..., image_patch=...)` |
| Demo patch from prompt text | `vision_patch_from_text` | `ai.vision_patch_from_text` |
| QA train uses input patch | `train_with_bpe` / `train` | `ai.Train` on vision bot |
| Safetensors `vision_patch_proj` key | export/import | `test_vision_patch_encoder_py.py` |
| `bin` meta `vision_patch_dim` | `export_bin` | `safetensors_vision_patch_proj_roundtrip` (Rust) |
| Merge OR semantics + proj average | `merge_models` | `test_merge_vision_or.py` |
| Quantize preserves flag + proj | `quantize_model` | `test_vision_chatbot_py.py` |

## Example

```powershell
python examples/vision_chatbot.py
```

Build a demo patch from UTF-8 bytes (training uses the **input** string automatically):

```python
import magicmindnet as ai

patch = ai.vision_patch_from_text("photo of a cat")
bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, vision=True)
loss = bot.compute_loss("describe this", "a cat", image_patch=patch)
```

## Checkpoint keys

When `vision=true`, safetensors checkpoints include:

- **meta** `vision_patch_dim` (64)
- **tensor** `vision_patch_proj` with shape `[d_model, 64]`

## Merge

`merge(a, b)` sets `vision = a.vision || b.vision` and element-wise-averages `vision_patch_proj` when present on either side. See [checkpoints.md](checkpoints.md).

## Limitations

- Single-channel 8×8 patch surrogate (not real RGB images or conv patch embed yet).
- Corpus LM training does not attach patches (text-only); QA training uses `vision_patch_from_text(input)`.
- No cross-attention between image and text streams yet.
- `DatasetImageGen` / `DatasetImageEdit` remain diffusion stubs, not `Chatbot` training.

## Roadmap

1. **Real patch embed** — `Conv2d` over RGB tiles instead of flat byte surrogate.
2. **Cross-attn block** — Q from text, K/V from image patches after self-attn.
3. **DatasetQA image paths** — load files from disk into normalized patches.

See also [training.md](training.md), [checkpoint_coverage.md](checkpoint_coverage.md), and [limitations.md](limitations.md).
