# Feature parity matrix

MagicMindNet aims for practical parity with common surfaces from **PyTorch**,
**llama.cpp**, and **Ollama**. This matrix tracks Wave 1–2 (done) and later waves
(planned).

| Feature | PyTorch | llama.cpp | Ollama | MagicMindNet |
|---------|---------|-----------|--------|--------------|
| typical_p sampling | transformers | `--typical` | — | **Wave 1** `typical_p=` |
| Mirostat v2 | — | `--mirostat 2` | — | **Wave 1** `mirostat=2` |
| Streaming tokens | generate iterators | server SSE | `/api/generate` stream | **Wave 1** `generate_stream` |
| Embeddings | `model.get_input_embeddings` / sentence | embedding endpoint | `/api/embeddings` | **Wave 1** `Chatbot.embed` |
| LR schedules | `CosineAnnealingLR` + warmup | — | — | **Wave 1** `TrainConfig(lr_schedule="cosine")` |
| weight_decay on TrainConfig | `AdamW(weight_decay=…)` | — | — | **Wave 1** `weight_decay=` |
| Chat messages / ChatML | `apply_chat_template` | — | `/api/chat` | **Wave 1** `format_chat_messages` / `chat_messages` |
| top_k / top_p / min_p | yes | yes | yes | done (pre-Wave 1) |
| KV-cache generate | — | yes | yes | done (pre-Wave 1) |
| GGUF load | — | native | yes | done (interop) |
| Grammar / JSON mode | Outlines / etc. | grammars | format | **Wave 2** `json_mode=` / `grammar=` |
| OpenAI-compatible HTTP | — | llama-server | Ollama API | **Wave 2** `OpenAIServer` |
| Tool / function calling | OpenAI tools | — | tools | **planned Wave 3** |
| Speculative decoding | — | draft models | — | **planned Wave 3** |
| Continuous batching server | vLLM | llama-server | Ollama | **planned Wave 3** |

## Wave 1 API sketch

```python
import magicmindnet as ai

bot = ai.Chatbot(vocab_size=512, n_layer=2, d_model=64, seed=1)

# llama.cpp-style sampling
bot.generate("hi", temperature=0.8, typical_p=0.9)
bot.generate("hi", temperature=0.8, mirostat=2, mirostat_tau=5.0, mirostat_eta=0.1)

# Ollama-style stream + embed
chunks = bot.generate_stream("hi", max_new_tokens=8, temperature=0.0)
vec = bot.embed("hello")  # list[float] length == d_model

# ChatML
prompt = ai.format_chat_messages([{"role": "user", "content": "Hi"}])
reply = bot.chat_messages([{"role": "user", "content": "Hi"}], max_new_tokens=16)

# PyTorch-style schedule
cfg = ai.TrainConfig(
    epochs=2,
    learning_rate=0.05,
    weight_decay=0.01,
    lr_schedule="cosine",
    warmup_steps=1,
    optimizer="adamw",
)
```

## Wave 2 API sketch

```python
import magicmindnet as ai
from magicmindnet.serve import OpenAIServer

bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=32, seed=11)

# Constrained decoding (llama.cpp / Ollama-style)
js = bot.generate('Return JSON: {"ok": true}', max_new_tokens=24, json_mode=True)
digits = bot.generate("n=", max_new_tokens=6, grammar="digit")

# Local OpenAI-compatible server
bot.save("tiny.mmn")
server = OpenAIServer(model_path="tiny.mmn", host="127.0.0.1", port=0)
base = server.start()  # e.g. http://127.0.0.1:54321
# POST {base}/v1/chat/completions , /v1/embeddings ; GET /v1/models , /health
server.stop()
```

Tests: `tests/test_feature_parity_py.py`, `tests/test_feature_parity_wave2_py.py`.
Eval tasks: `gen_typical_p_smoke`, `stream_generate`, `embed_mean_pool`,
`json_mode_smoke` (optional).
