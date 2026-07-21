# Universal hub loading (`ai.from_pretrained`)

Load **any** model from Hugging Face Hub, ModelScope, Ollama, or a local path, then
`generate` / `predict` / `finetune` / `train` / `save` through one API.

```python
import magicmindnet as ai

model = ai.from_pretrained("org/name")          # Hugging Face
model = ai.from_pretrained("hf://org/name")
model = ai.from_pretrained("ms://org/name")     # ModelScope (pip install modelscope)
model = ai.from_pretrained("ollama://llama3.2") # local Ollama daemon
model = ai.from_pretrained("./my-checkpoint.mmn")
```

## Routing (by layout, not by repo name)

| Signal | Family | Backend |
|--------|--------|---------|
| `*.gguf` | `gguf` / causal-lm | **native** `Chatbot` (`ai.load`) when shapes adapt |
| Causal LM safetensors (`*ForCausalLM`) | `causal-lm` | transformers (native adapt when possible) |
| `*ForSequenceClassification` / rerankers | `classifier` / `reranker` | transformers pipeline |
| Encoder–decoder (`M2M100`, T5, …) | `seq2seq` | transformers model+tokenizer |
| `model_index.json` / Diffusers | `diffusion` / `video` | diffusers |
| `text-to-speech` | `tts` | transformers / foreign |
| Ollama id | `causal-lm` | Ollama HTTP API |

This is intentionally **not** a list of special-cased repos. New Hub models route by
`pipeline_tag`, `architectures`, and files.

## Train / finetune

```python
# Native Chatbot / Classifier
model.finetune(ai.DatasetCorpus(data=["…"] * 32), epochs=1)
model.finetune(ai.DatasetClassification(data=[{"text": "…", "label": "joy"}]), epochs=1)

# Same methods on HubModel wrappers (transformers Trainer under the hood)
```

`DatasetClassification.as_pairs()` and `DatasetCorpus.as_texts()` expose rows for custom loops.

## Optional dependencies

| Feature | Package |
|---------|---------|
| HF download | `huggingface_hub` |
| Foreign LM / classifier / seq2seq | `transformers`, `torch` |
| Diffusers / SD / Wan | `diffusers`, `torch` |
| ModelScope | `modelscope` |
| Ollama | running `ollama` daemon |

Native GGUF / MMN checkpoints need **no** extras beyond MagicMindNet itself.

## Hands-on matrix (verified this branch)

| Model | Load | Infer | Finetune |
|-------|------|-------|----------|
| Local `.mmn` Chatbot | OK | generate | OK |
| `j-hartmann/emotion-english-distilroberta-base` | OK | predict_label | OK (Trainer) |
| `BAAI/bge-reranker-v2-m3` | OK | predict | routed |
| `facebook/nllb-200-distilled-600M` | OK | generate (seq2seq) | routed |
| `Qwen/Qwen3-0.6B` | OK | generate | OK (Trainer) |
| `Qwen/Qwen3-0.6B-GGUF` | OK native (`head_dim=128`) | generate | native train |
| `mlx-community/Qwen3-0.6B-4bit` | OK (transformers) | generate | routed |
| `CompVis/stable-diffusion-v1-4` | OK diffusers | pipeline | foreign |
| `sd2-community/stable-diffusion-2-inpainting` | OK diffusers | pipeline | foreign |
| `Wan-AI/Wan2.1-T2V-1.3B(-Diffusers)` | OK route `video` | lazy weights | foreign |
| `hexgrad/Kokoro-82M` | OK route `tts` | foreign | foreign |
| `ollama://…` | OK route | needs daemon | — |

Script: `python scripts/hub_hands_on.py`.

## Limitations

- Native Chatbot still assumes transformer-decoder math (GELU/SwiGLU, LN/RMS, optional
  independent `head_dim` for Qwen-style GQA). Exotic blocks (video DiT, StyleTTS) stay
  on the foreign backend.
- Full multi-GB Diffusers weight downloads are lazy; configs route immediately.
- Ollama generate requires a reachable `OLLAMA_HOST` (default `http://127.0.0.1:11434`).
