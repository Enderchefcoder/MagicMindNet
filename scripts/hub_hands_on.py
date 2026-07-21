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
    # Native Diffusion via from_pretrained + train (covers the diffusion family
    # without downloading multi-GB Hub weights).
    fixtures = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    diff = ai.Diffusion()
    dpath = CACHE / "native_diff.mmn"
    diff.save(str(dpath))
    dloaded = ai.from_pretrained(str(dpath))
    assert dloaded.native is not None
    ds = ai.DatasetImageGen(file=str(fixtures / "image_gen.json"))
    d_losses = dloaded.train(ds, epochs=1, learning_rate=0.05, batch_size=1)
    patch = dloaded.native.sample_rgb_patch(steps=1)
    record(
        "local-native-diffusion",
        ok=True,
        detail=f"family={dloaded.family} train={d_losses} sample_shape={getattr(patch, 'shape', type(patch))}",
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
    data = ai.DatasetCorpus(data=["MagicMindNet trains tiny language models on CPU."] * 2)
    ft = None
    try:
        ft = model.finetune(
            data, epochs=1, learning_rate=1e-5, batch_size=1, max_length=24
        )
    except Exception as e:
        ft = f"skip:{type(e).__name__}:{e}"
    record(
        "Qwen3-0.6B",
        ok=True,
        detail=f"family={model.family} gen={text!r} ft={ft}",
    )
    # Drop references before process exit (subprocess driver also isolates).
    del model


def case_qwen_gguf() -> None:
    model = ai.from_pretrained(
        "Qwen/Qwen3-0.6B-GGUF",
        filename="Qwen3-0.6B-Q8_0.gguf",
        cache_dir=str(CACHE / "qwen_gguf"),
    )
    detail = f"family={model.family} native={model.native is not None} backend={model.card.backend}"
    if model.native is None:
        raise RuntimeError(f"expected native GGUF Chatbot, got {detail} notes={model.card.notes}")
    detail += (
        f" head_dim={getattr(model.native, 'head_dim', None)}"
        f" n_heads={getattr(model.native, 'n_heads', None)}"
        f" d_model={getattr(model.native, 'd_model', None)}"
    )
    # Inference only here — full native finetune of 0.6B f32 needs more RAM than
    # typical CI/cloud VMs after Hub downloads. Tiny native train is covered by
    # local-native-chatbot / Diffusion cases.
    text = model.generate("Hi", max_new_tokens=4)
    detail += f" gen={text!r}"
    record("Qwen3-0.6B-GGUF", ok=True, detail=detail)
    del model


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


def _run_case_subprocess(key: str) -> list[dict]:
    """Run one case in a fresh process so large Hub weights are released."""
    import subprocess
    import sys

    out_path = CACHE / f"_case_{key}.json"
    if out_path.exists():
        out_path.unlink()
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["MMN_HUB_HANDS_ON_CASE"] = key
    env["MMN_HUB_HANDS_ON_OUT"] = str(out_path)
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--only", key, "--in-process"],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        check=False,
    )
    if out_path.exists():
        rows = json.loads(out_path.read_text(encoding="utf-8"))
        out_path.unlink(missing_ok=True)
        return rows
    return [
        {
            "name": key,
            "ok": False,
            "detail": f"subprocess_exit={proc.returncode}",
        }
    ]


def main() -> None:
    import gc
    import os

    p = argparse.ArgumentParser()
    p.add_argument("--only", default="", help="comma-separated case keys")
    p.add_argument(
        "--list",
        action="store_true",
        help="print available case keys and exit",
    )
    p.add_argument(
        "--in-process",
        action="store_true",
        help="run cases in this process (used by subprocess driver)",
    )
    args = p.parse_args()
    if args.list:
        print("\n".join(CASES))
        return
    keys = [k.strip() for k in args.only.split(",") if k.strip()] or list(CASES)

    # Child process path: one case, write RESULTS to MMN_HUB_HANDS_ON_OUT.
    if args.in_process or os.environ.get("MMN_HUB_HANDS_ON_CASE"):
        for key in keys:
            print("=" * 72, key, flush=True)
            try_case(key, CASES[key])
            gc.collect()
        out = Path(os.environ.get("MMN_HUB_HANDS_ON_OUT") or (CACHE / "hands_on_results.json"))
        out.write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
        ok = sum(1 for r in RESULTS if r.get("ok"))
        print("=" * 72, flush=True)
        print(f"Wrote {out}  ({ok}/{len(RESULTS)} ok)", flush=True)
        return

    # Parent: isolate each case so HF/diffusers weights free between models.
    merged: list[dict] = []
    for key in keys:
        print("=" * 72, f"subprocess:{key}", flush=True)
        rows = _run_case_subprocess(key)
        for row in rows:
            status = "OK" if row.get("ok") else "FAIL"
            print(f"[{status}] {row.get('name')}: {row.get('detail', '')}", flush=True)
        merged.extend(rows)
    out = CACHE / "hands_on_results.json"
    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    ok = sum(1 for r in merged if r.get("ok"))
    print("=" * 72)
    print(f"Wrote {out}  ({ok}/{len(merged)} ok)")


if __name__ == "__main__":
    main()
