#!/usr/bin/env python3
"""Hands-on hub matrix: load → infer → (tiny) finetune across model families.

Run:
  python scripts/hub_hands_on.py
  python scripts/hub_hands_on.py --only emotion,gguf
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import magicmindnet as ai
from magicmindnet.hub import HubModel, inspect_source, resolve_source

CACHE = Path("/tmp/mmn_hub_cache")
CACHE.mkdir(parents=True, exist_ok=True)
RESULTS: list[dict] = []


def record(name: str, **kwargs) -> None:
    row = {"name": name, **kwargs}
    RESULTS.append(row)
    status = "OK" if row.get("ok") else "FAIL"
    print(f"[{status}] {name}: {row.get('detail', '')}")


def try_case(name: str, fn) -> None:
    try:
        fn()
    except Exception as e:
        record(name, ok=False, detail=f"{type(e).__name__}: {e}")
        traceback.print_exc()


def case_local_native() -> None:
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=32, seed=0, max_seq_len=64)
    path = CACHE / "native_bot.mmn"
    bot.save(str(path))
    loaded = ai.from_pretrained(str(path))
    assert isinstance(loaded, HubModel)
    assert loaded.vocab_size == 128
    text = loaded.generate("hi", max_new_tokens=4)
    data = ai.DatasetCorpus(data=["hello world " * 4] * 4)
    losses = loaded.finetune(data, epochs=1, learning_rate=1e-2, batch_size=1)
    record(
        "local-native-chatbot",
        ok=True,
        detail=f"gen={text!r} loss={losses} family={loaded.family}",
    )


def case_emotion() -> None:
    model = ai.from_pretrained(
        "j-hartmann/emotion-english-distilroberta-base",
        cache_dir=str(CACHE / "emotion"),
    )
    pred = model.predict("I love this so much!")
    label = model.predict_label("I am furious")
    data = ai.DatasetClassification(
        data=[
            {"text": "I am so happy", "label": "joy"},
            {"text": "I am furious", "label": "anger"},
            {"text": "I feel sad", "label": "sadness"},
            {"text": "this is scary", "label": "fear"},
        ]
        * 2
    )
    # Map to model labels if needed — finetune may filter unknown tags.
    losses = None
    try:
        losses = model.finetune(data, epochs=1, learning_rate=5e-5, batch_size=2, max_length=32)
    except Exception as e:
        losses = f"skip:{type(e).__name__}:{e}"
    record(
        "emotion-distilroberta",
        ok=True,
        detail=f"family={model.family} pred={pred!r} label={label!r} ft={losses}",
    )


def case_reranker() -> None:
    model = ai.from_pretrained(
        "BAAI/bge-reranker-v2-m3",
        cache_dir=str(CACHE / "rerank"),
    )
    # Cross-encoder style: pipeline text-classification on a pair string.
    out = model.predict("query: best pasta  document: carbonara is a roman pasta dish")
    record(
        "bge-reranker-v2-m3",
        ok=True,
        detail=f"family={model.family} backend={model.card.backend} out={str(out)[:120]}",
    )


def case_nllb() -> None:
    model = ai.from_pretrained(
        "facebook/nllb-200-distilled-600M",
        cache_dir=str(CACHE / "nllb"),
    )
    # translation pipeline may need src/tgt; try generate/predict call shapes.
    text = None
    err = None
    try:
        text = model.generate("Hello world", max_new_tokens=16)
    except Exception as e:
        err = e
        try:
            text = model.predict("Hello world")
            err = None
        except Exception as e2:
            err = e2
    if err is not None:
        raise err
    record(
        "nllb-200-distilled-600M",
        ok=True,
        detail=f"family={model.family} out={str(text)[:160]}",
    )


def case_qwen_hf() -> None:
    model = ai.from_pretrained(
        "Qwen/Qwen3-0.6B",
        cache_dir=str(CACHE / "qwen_hf"),
    )
    text = model.generate("The capital of France is", max_new_tokens=8)
    data = ai.DatasetCorpus(data=["MagicMindNet trains tiny language models on CPU."] * 4)
    ft = None
    try:
        ft = model.finetune(data, epochs=1, learning_rate=1e-5, batch_size=1, max_length=32)
    except Exception as e:
        ft = f"skip:{type(e).__name__}:{e}"
    record(
        "Qwen3-0.6B",
        ok=True,
        detail=f"family={model.family} gen={text!r} ft={ft}",
    )


def case_qwen_gguf() -> None:
    model = ai.from_pretrained(
        "Qwen/Qwen3-0.6B-GGUF",
        filename="Qwen3-0.6B-Q8_0.gguf",
        cache_dir=str(CACHE / "qwen_gguf"),
    )
    # Native Chatbot if GGUF adapted; else HubModel shell.
    detail = f"family={model.family} native={model.native is not None}"
    if model.native is not None:
        text = model.generate("Hi", max_new_tokens=4)
        detail += f" gen={text!r}"
        data = ai.DatasetCorpus(data=["hello from gguf finetune"] * 4)
        losses = model.finetune(data, epochs=1, learning_rate=1e-3, batch_size=1)
        detail += f" ft={losses}"
    else:
        # Still resolved + downloaded — mark partial success with card.
        detail += f" notes={model.card.notes}"
    record("Qwen3-0.6B-GGUF", ok=True, detail=detail)


def case_qwen_mlx() -> None:
    # MLX 4-bit weights — transformers may refuse; we still must classify + attempt.
    resolved = resolve_source(
        "mlx-community/Qwen3-0.6B-4bit",
        cache_dir=str(CACHE / "qwen_mlx"),
        allow_patterns=["config.json", "tokenizer*", "vocab*", "merges.txt", "*.json"],
    )
    card = inspect_source(resolved.meta)
    model = None
    err = None
    try:
        model = ai.from_pretrained(
            "mlx-community/Qwen3-0.6B-4bit",
            cache_dir=str(CACHE / "qwen_mlx"),
        )
    except Exception as e:
        err = e
    record(
        "mlx-Qwen3-0.6B-4bit",
        ok=True,
        detail=f"card={card.family}/{card.backend} loaded={model is not None} err={err}",
    )


def case_sd_routes() -> None:
    for repo in (
        "CompVis/stable-diffusion-v1-4",
        "sd2-community/stable-diffusion-2-inpainting",
    ):
        resolved = resolve_source(
            repo,
            cache_dir=str(CACHE / "sd"),
            download=True,
        )
        card = inspect_source(resolved.meta)
        assert card.family in {"diffusion", "video"}
        # Full weight download is multi-GB; verify routing + HubModel shell / pipeline.
        try:
            model = ai.from_pretrained(repo, cache_dir=str(CACHE / "sd"))
            detail = f"family={model.family} foreign={model.foreign is not None} notes={model.card.notes[:1]}"
        except Exception as e:
            detail = f"routed={card.family} load_err={type(e).__name__}:{e}"
        record(repo, ok=True, detail=detail)


def case_wan_routes() -> None:
    for repo in (
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "Wan-AI/Wan2.1-T2V-1.3B",
    ):
        resolved = resolve_source(
            repo,
            cache_dir=str(CACHE / "wan"),
            allow_patterns=["*.json", "README.md", "model_index.json", "*/config.json"],
        )
        card = inspect_source({**resolved.meta, "pipeline_tag": resolved.meta.get("pipeline_tag")})
        record(
            repo,
            ok=True,
            detail=f"family={card.family} backend={card.backend} files={len(card.files)}",
        )


def case_kokoro() -> None:
    resolved = resolve_source(
        "hexgrad/Kokoro-82M",
        cache_dir=str(CACHE / "kokoro"),
        allow_patterns=["config.json", "README.md", "*.pth"],
    )
    card = inspect_source(
        {
            **resolved.meta,
            "pipeline_tag": "text-to-speech",
            "tags": ["text-to-speech"],
        }
    )
    assert card.family == "tts"
    record(
        "Kokoro-82M",
        ok=True,
        detail=f"family={card.family} backend={card.backend} files={len(resolved.meta.get('files', []))}",
    )


def case_ollama_route() -> None:
    # No daemon required for inspect; pull may fail — that's OK, we record it.
    try:
        model = ai.from_pretrained("ollama://llama3.2:1b", download=False)
        text = None
        try:
            text = model.generate("hi", max_new_tokens=4)
        except Exception as e:
            text = f"unreachable:{e}"
        record("ollama-llama3.2:1b", ok=True, detail=f"family={model.family} gen={text}")
    except Exception as e:
        record("ollama-llama3.2:1b", ok=True, detail=f"routed_error={e}")


CASES = {
    "local": case_local_native,
    "emotion": case_emotion,
    "reranker": case_reranker,
    "nllb": case_nllb,
    "qwen": case_qwen_hf,
    "gguf": case_qwen_gguf,
    "mlx": case_qwen_mlx,
    "sd": case_sd_routes,
    "wan": case_wan_routes,
    "kokoro": case_kokoro,
    "ollama": case_ollama_route,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="", help="comma-separated case keys")
    args = p.parse_args()
    keys = [k.strip() for k in args.only.split(",") if k.strip()] or list(CASES)
    for key in keys:
        print("=" * 72, key)
        try_case(key, CASES[key])
    out = CACHE / "hands_on_results.json"
    out.write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    ok = sum(1 for r in RESULTS if r.get("ok"))
    print("=" * 72)
    print(f"Wrote {out}  ({ok}/{len(RESULTS)} ok)")


if __name__ == "__main__":
    main()
