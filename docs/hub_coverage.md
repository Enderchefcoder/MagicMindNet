# Hub coverage matrix

Offline / hands-on regression for `ai.from_pretrained` and `HubModel`.

| Capability | Offline test | Hands-on / network |
|------------|--------------|--------------------|
| Export surface (`from_pretrained`, `HubModel`, `ModelCard`, `inspect_source`, `resolve_source`, `list_hub_families`) | `test_hub_public_exports`, `test_from_pretrained_exported` | — |
| Spec prefixes (`hf://`, `ms://`, ModelScope URL, `ollama://`, local) | `test_parse_spec_*` | — |
| Inspect families (causal/cls/rerank/seq2seq/diffusion/video/tts/asr/embedding/zero-shot/fill-mask/QA/vlm/arrays/MLX) | `test_inspect_*` | — |
| Native Chatbot generate + finetune via hub | `test_hub_model_finetune_causal_lm_api`, `hub_local_roundtrip` | `hub_hands_on --only local` |
| Native Classifier finetune + predict | `test_hub_model_finetune_classifier_api` | emotion live (`@pytest.mark.network`) |
| Native Diffusion generate + finetune | `test_synthetic_diffusion_generate_and_finetune` | local diffusion in hands-on |
| Rerank `score_pairs` / `rerank` | `test_synthetic_reranker_*` | bge-reranker (predict) |
| Ollama `/api/generate` + `/api/chat` | urllib mocks | daemon optional |
| Foreign Trainer finetune | Trainer mock | Qwen / emotion live |
| GGUF / video hollow-shell errors | clear `RuntimeError` tests | Qwen GGUF native |
| Directory detect (`model_index.json`, nested `.mmn`) | Rust `detect.rs` | SD route |
| CI smoke | `smoke_examples.sh` hub local + examples | — |

Full API: [hub.md](hub.md). Hands-on: `python scripts/hub_hands_on.py`.
