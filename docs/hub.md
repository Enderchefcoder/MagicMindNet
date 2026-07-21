# Universal hub loading (`ai.from_pretrained`)

Load **any** model from Hugging Face Hub, ModelScope, Ollama, or a local path, then
`generate` / `predict` / `finetune` / `train` / `save` through one API.

```python
import magicmindnet as ai

model = ai.from_pretrained("org/name")          # Hugging Face
model = ai.from_pretrained("hf://org/name")
model = ai.from_pretrained("ms://org/name")     # ModelScope (pip install -e ".[hub]")
model = ai.from_pretrained("ollama://llama3.2") # local Ollama daemon
model = ai.from_pretrained("./my-checkpoint.mmn")
print(model.capabilities())
print(model.generate("hi", max_new_tokens=16))
```

## Routing (by layout, not by repo name)

| Signal | Family | Backend |
|--------|--------|---------|
| `*.gguf` | `gguf` / causal-lm | **native** `Chatbot` when shapes adapt |
| Causal LM safetensors (`*ForCausalLM`) | `causal-lm` | transformers (native adapt when possible) |
| `*ForSequenceClassification` / rerankers | `classifier` / `reranker` | transformers pipeline; `score_pairs` / `rerank` |
| Encoder–decoder (`M2M100`, T5, …) | `seq2seq` | transformers model+tokenizer |
| `model_index.json` / Diffusers | `diffusion` / `video` | diffusers (`generate` → pipe / frames) |
| `text-to-speech` / ASR | `tts` / `asr` | transformers pipelines |
| feature-extraction / sentence-similarity | `embedding` | `embed()` |
| zero-shot / fill-mask / QA | matching family | transformers pipelines |
| image-text-to-text | `vlm` | transformers |
| Ollama id | `causal-lm` | `/api/generate` + `/api/chat` |

This is intentionally **not** a list of special-cased repos.

## Route vs infer vs finetune

| Family | Route | Infer | Finetune |
|--------|-------|-------|----------|
| Native `.mmn` / GGUF Chatbot | yes | `generate` / `chat` | `finetune` / `train` |
| Classifier / reranker | yes | `predict` / `score_pairs` / `rerank` | Trainer (seq-cls) |
| Seq2seq (NLLB, …) | yes | `generate` | causal-style Trainer when texts available |
| Diffusion (SD, …) | yes | `generate` → pipe / native `sample_rgb_patch` | native `DatasetImageGen` / foreign deferred |
| Video (Wan, …) | yes | needs loaded pipeline (multi-GB) | foreign / native toy Diffusion |
| TTS / ASR | yes | pipeline `generate` / `predict` | foreign (not native) |
| Embedding | yes | `embed` / feature-extraction | — |
| Ollama | yes | generate + chat | — |

Coverage matrix: [hub_coverage.md](hub_coverage.md).

## Train / finetune

```python
model.finetune(ai.DatasetCorpus(data=["…"] * 32), epochs=1)
model.finetune(ai.DatasetClassification(data=[{"text": "…", "label": "joy"}]), epochs=1)
model.finetune(ai.DatasetQA(data=[{"input": "q", "output": "a"}]), epochs=1)  # as_pairs
model.score_pairs("query", ["doc a", "doc b"])
model.rerank("query", ["doc a", "doc b"], top_k=5)
```

`DatasetClassification.as_pairs()`, `DatasetCorpus.as_texts()`, and `DatasetQA.as_pairs()`
expose rows for custom loops.

## Optional dependencies

```bash
pip install -e ".[hub]"   # huggingface_hub, transformers, diffusers, torch, modelscope, …
```

| Feature | Package |
|---------|---------|
| HF download | `huggingface_hub` |
| Foreign LM / classifier / seq2seq / TTS / ASR | `transformers`, `torch` |
| Diffusers / SD / Wan | `diffusers`, `torch` |
| ModelScope | `modelscope` |
| Ollama | running `ollama` daemon |

Native GGUF / MMN checkpoints need **no** extras.

## Hands-on

```bash
python scripts/hub_hands_on.py
python scripts/hub_hands_on.py --only local,emotion,gguf
python examples/hub_local_roundtrip.py
```

## Limitations

- Native Chatbot still assumes transformer-decoder math (GELU/SwiGLU, LN/RMS, optional
  independent `head_dim` for Qwen-style GQA). Exotic blocks (video DiT, StyleTTS) stay
  on the foreign backend.
- Full multi-GB Diffusers / Wan weight downloads are lazy; configs route immediately.
  Video `generate` without a loaded pipe raises a clear error.
- Failed GGUF native adapt raises (no silent empty shell).
- Ollama requires a reachable `OLLAMA_HOST` (default `http://127.0.0.1:11434`).
- LoopLoRA QKV adapters (`lora_rank`) and `final_norm` are implemented on native Chatbot (see [API.md](API.md), [limitations.md](limitations.md)).
