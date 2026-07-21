#!/usr/bin/env python3
"""Print hub family routing catalog (offline) + optional local from_pretrained."""

from __future__ import annotations

import magicmindnet as ai

# Layout-driven families (not a hard-coded list of every Hub repo).
CATALOG = [
    ("gguf / .mmn", "Native Chatbot when tensor shapes adapt"),
    ("causal-lm", "HF ForCausalLM · transformers fallback"),
    ("classifier / reranker", "seq-cls · score_pairs / rerank"),
    ("seq2seq", "encoder–decoder generate"),
    ("diffusion / video", "Diffusers pipes"),
    ("embedding", "feature-extraction / sentence-similarity"),
    ("asr / tts", "speech pipelines"),
    ("vlm", "image-text-to-text"),
    ("ollama://", "local Ollama daemon"),
]


def main() -> None:
    print(f"magicmindnet {ai.__version__} — hub family catalog")
    for name, desc in CATALOG:
        print(f"  • {name:22}  {desc}")

    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=32, seed=3)
    path = "_hub_catalog_local.mmn"
    bot.save(path)
    model = ai.from_pretrained(path)
    caps = model.capabilities() if hasattr(model, "capabilities") else {}
    print(f"local from_pretrained caps={caps}")
    print("hub_catalog ok — see docs/hub.md for routing rules")
    print("Tip: pip install 'magicmindnet[hub]' then ai.from_pretrained('org/name')")


if __name__ == "__main__":
    main()
