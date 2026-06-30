# Vision chatbot path

The `Chatbot(vision=True)` flag is **checkpoint metadata** today: it records intent for multimodal models and flows through IO, merge, and training — but there is no separate vision encoder or image forward path yet.

## What works (alpha)

| Behavior | Rust | Python |
|----------|------|--------|
| `has_vision` getter | `Chatbot::has_vision` | `bot.has_vision` |
| Safetensors roundtrip | export/import meta | `test_safetensors_vision_roundtrip.py` |
| `bin` / empty defaults | `export_bin` (incl. optional `use_learned_pos_embed`, `max_seq_len`) | `test_bin_shape_getters.py`, `test_bin_roundtrip_preserves_learned_pos_embed_meta` |
| Merge OR semantics | `merge_models` | `test_merge_vision_or.py` |
| Train QA + keep flag | `vision_chatbot_trains_and_keeps_vision_flag` | `test_vision_chatbot_py.py` |
| Quantize preserves flag | `quantize_preserves_vision_flag_on_export_roundtrip` | `test_vision_chatbot_py.py` |
| `repr` shows vision | — | `test_vision_chatbot_py.py` |

## Example

```powershell
python examples/vision_chatbot.py
```

## Merge

`merge(a, b)` sets `vision = a.vision || b.vision` (element-wise weight mean unchanged). See [checkpoints.md](checkpoints.md).

## Limitations

- No image tensors in `forward_hidden` / `forward_logits`.
- `DatasetImageGen` / `DatasetImageEdit` are for diffusion stubs, not `Chatbot` training.
- Production vision would need a patch encoder + cross-attn (see **Roadmap** below).

## Roadmap: cross-attention sketch (`vision=True`)

Target architecture for a future vision-capable `Chatbot`:

1. **Patch encoder** — `Conv2d` or linear patch embed maps image `[C,H,W]` → `[n_patches, d_model]` (new `mmn-nn` module).
2. **Dual stream** — text tokens keep causal self-attn; image patches use full (non-causal) self-attn or frozen encoder.
3. **Cross-attn block** — after text self-attn, optional `CrossAttention` where Q from text hidden, K/V from image patches (bidirectional over patches).
4. **Checkpoint keys** — `vision_encoder.*`, `blocks.{i}.cross_attn.{q,k,v,out}` under `mmn-safetensors-v1` with `vision: true` meta.
5. **Train path** — `DatasetQA` rows with image paths (future) or paired `DatasetClassification` features; RL/SPIN unchanged on policy head.

Until then, `vision=True` remains metadata-only. Tracked in [limitations.md](limitations.md).

See also [training.md](training.md) and [checkpoint_coverage.md](checkpoint_coverage.md).
