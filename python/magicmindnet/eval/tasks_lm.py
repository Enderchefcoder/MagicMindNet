"""Language-model, Glint, GQA, BPE/PE/RoPE, and generate eval tasks."""

from __future__ import annotations

import json
import time
from pathlib import Path

import magicmindnet as ai
from magicmindnet.eval._util import (
    corpus_dataset,
    ensure_work,
    qa_dataset,
    tiny_chatbot,
    train_cfg,
)
from magicmindnet.eval.registry import register
from magicmindnet.eval.types import Metric, TaskResult


def _loss_train_result(
    name: str,
    bot: ai.Chatbot,
    ds: ai.DatasetQA | ai.DatasetCorpus,
    *,
    epochs: int = 3,
    bpe: ai.BytePairEncoder | None = None,
    meta: dict | None = None,
) -> TaskResult:
    t0 = time.perf_counter()
    before = float(bot.compute_mean_loss(ds, bpe_encoder=bpe))
    ai.Train(bot, ds, train_cfg(epochs=epochs), bpe_encoder=bpe)
    after = float(bot.compute_mean_loss(ds, bpe_encoder=bpe))
    return TaskResult(
        name=name,
        ok=after <= before + 1e-6,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metrics=[
            Metric("loss_before", before, higher_is_better=False),
            Metric("loss_after", after, higher_is_better=False),
            Metric("loss_delta", after - before, higher_is_better=False),
        ],
        meta=meta or {},
    )


@register("lm_qa_loss", suites=("smoke", "lm"), description="QA mean CE loss (no train)")
def lm_qa_loss(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del work_dir
    t0 = time.perf_counter()
    bot = tiny_chatbot(seed)
    loss = float(bot.compute_mean_loss(qa_dataset()))
    return TaskResult(
        name="lm_qa_loss",
        ok=loss > 0 and loss == loss,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metrics=[Metric("mean_loss", loss, higher_is_better=False)],
    )


@register("lm_qa_train", suites=("smoke", "lm", "train"), description="QA train loss delta")
def lm_qa_train(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del work_dir
    return _loss_train_result("lm_qa_train", tiny_chatbot(seed), qa_dataset(), epochs=3)


@register("lm_corpus_loss", suites=("lm",), description="Corpus mean CE loss")
def lm_corpus_loss(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del work_dir
    t0 = time.perf_counter()
    bot = tiny_chatbot(seed + 2)
    loss = float(bot.compute_mean_loss(corpus_dataset()))
    return TaskResult(
        name="lm_corpus_loss",
        ok=loss > 0 and loss == loss,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metrics=[Metric("mean_loss", loss, higher_is_better=False)],
    )


@register("lm_corpus_train", suites=("lm", "train"), description="Corpus train loss delta")
def lm_corpus_train(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del work_dir
    return _loss_train_result(
        "lm_corpus_train",
        tiny_chatbot(seed + 2),
        corpus_dataset(),
        epochs=3,
    )


@register(
    "lm_learned_pe_train",
    suites=("lm", "train"),
    description="Learned positional embedding train delta",
)
def lm_learned_pe_train(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del work_dir
    bot = tiny_chatbot(seed, use_learned_pos_embed=True, max_seq_len=128)
    return _loss_train_result(
        "lm_learned_pe_train",
        bot,
        qa_dataset(),
        epochs=3,
        meta={"use_learned_pos_embed": True},
    )


@register("lm_rope_train", suites=("lm", "train"), description="RoPE train loss delta")
def lm_rope_train(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del work_dir
    bot = tiny_chatbot(seed, use_rope=True)
    return _loss_train_result(
        "lm_rope_train",
        bot,
        qa_dataset(),
        epochs=3,
        meta={"use_rope": True, "rope_theta": bot.rope_theta},
    )


@register("lm_bpe_train", suites=("lm", "train"), description="BPE-tokenized QA train delta")
def lm_bpe_train(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del work_dir
    vocab = 512
    bot = tiny_chatbot(seed, vocab_size=vocab)
    texts = ["repeat repeat repeat token"] * 12
    bpe = ai.BytePairEncoder.train(texts, vocab_size=vocab, num_merges=24)
    return _loss_train_result(
        "lm_bpe_train",
        bot,
        qa_dataset(),
        epochs=4,
        bpe=bpe,
        meta={"bpe_merges": bpe.merge_count, "vocab_size": vocab},
    )


@register("lm_gqa_train", suites=("lm", "train"), description="GQA (n_kv_heads < n_heads) train")
def lm_gqa_train(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del work_dir
    bot = tiny_chatbot(seed, d_model=64, n_heads=4, n_kv_heads=2)
    return _loss_train_result(
        "lm_gqa_train",
        bot,
        qa_dataset(),
        epochs=3,
        meta={"n_heads": bot.n_heads, "n_kv_heads": bot.n_kv_heads},
    )


@register(
    "lm_glint_train",
    suites=("smoke", "lm", "glint", "train"),
    description="Glint-style (loops/RMS/SwiGLU/tied) train delta",
)
def lm_glint_train(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del work_dir
    bot = tiny_chatbot(
        seed,
        n_loops=2,
        norm="rms",
        ffn="swiglu",
        tie_embeddings=True,
        loop_embed=True,
        final_norm=True,
        lora_rank=2,
    )
    return _loss_train_result(
        "lm_glint_train",
        bot,
        qa_dataset(),
        epochs=3,
        meta={
            "n_loops": bot.n_loops,
            "norm": bot.norm,
            "ffn": bot.ffn,
            "tie_embeddings": bot.tie_embeddings,
            "loop_embed": bot.loop_embed,
            "final_norm": bot.final_norm,
            "lora_rank": bot.lora_rank,
        },
    )


@register(
    "lm_generate_latency",
    suites=("smoke", "lm", "generate"),
    description="Chatbot.generate wall time",
)
def lm_generate_latency(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del work_dir
    bot = tiny_chatbot(seed, n_layer=1)
    # Warmup
    bot.generate("hi", max_new_tokens=2)
    t0 = time.perf_counter()
    text = bot.generate("hello world", max_new_tokens=8)
    ms = (time.perf_counter() - t0) * 1000
    return TaskResult(
        name="lm_generate_latency",
        ok=isinstance(text, str) and len(text) >= 0 and ms >= 0,
        elapsed_ms=ms,
        metrics=[
            Metric("generate_ms", ms, unit="ms", higher_is_better=False),
            Metric("max_new_tokens", 8.0),
        ],
        meta={"preview": text[:64]},
    )


@register(
    "gen_typical_p_smoke",
    suites=("smoke", "generate"),
    description="generate with typical_p and mirostat kwargs",
)
def gen_typical_p_smoke(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del work_dir
    bot = tiny_chatbot(seed, n_layer=1)
    t0 = time.perf_counter()
    a = bot.generate("hi", max_new_tokens=4, temperature=0.8, typical_p=0.9)
    b = bot.generate("hi", max_new_tokens=4, temperature=0.8, mirostat=2)
    ms = (time.perf_counter() - t0) * 1000
    return TaskResult(
        name="gen_typical_p_smoke",
        ok=isinstance(a, str) and isinstance(b, str),
        elapsed_ms=ms,
        metrics=[Metric("generate_ms", ms, unit="ms", higher_is_better=False)],
    )


@register(
    "json_mode_smoke",
    suites=("smoke", "generate"),
    description="generate with json_mode yields object/array-shaped text",
)
def json_mode_smoke(*, seed: int = 11, work_dir: Path | None = None) -> TaskResult:
    del work_dir
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=32, seed=seed)
    t0 = time.perf_counter()
    out = bot.generate(
        'Return JSON: {"ok": true}',
        max_new_tokens=24,
        temperature=0.8,
        json_mode=True,
    )
    ms = (time.perf_counter() - t0) * 1000
    text = out.strip() if isinstance(out, str) else ""
    ok = False
    try:
        ok = (text.startswith("{") or text.startswith("[")) and json.loads(text) is not None
    except Exception:
        ok = False
    return TaskResult(
        name="json_mode_smoke",
        ok=ok,
        elapsed_ms=ms,
        metrics=[Metric("generate_ms", ms, unit="ms", higher_is_better=False)],
        meta={"preview": text[:64]},
    )


@register(
    "stream_generate",
    suites=("smoke", "generate"),
    description="generate_stream chunks join to generate",
)
def stream_generate(*, seed: int = 2, work_dir: Path | None = None) -> TaskResult:
    del work_dir
    bot = tiny_chatbot(seed, n_layer=1)
    t0 = time.perf_counter()
    chunks = bot.generate_stream("ab", max_new_tokens=5, temperature=0.0)
    full = bot.generate("ab", max_new_tokens=5, temperature=0.0)
    ms = (time.perf_counter() - t0) * 1000
    ok = isinstance(chunks, list) and "".join(chunks) == full
    return TaskResult(
        name="stream_generate",
        ok=ok,
        elapsed_ms=ms,
        metrics=[Metric("n_chunks", float(len(chunks)))],
    )


@register(
    "embed_mean_pool",
    suites=("smoke", "generate"),
    description="Chatbot.embed mean-pool d_model vector",
)
def embed_mean_pool(*, seed: int = 3, work_dir: Path | None = None) -> TaskResult:
    del work_dir
    bot = tiny_chatbot(seed, n_layer=1)
    t0 = time.perf_counter()
    v = bot.embed("hello")
    ms = (time.perf_counter() - t0) * 1000
    ok = isinstance(v, list) and len(v) == bot.d_model
    return TaskResult(
        name="embed_mean_pool",
        ok=ok,
        elapsed_ms=ms,
        metrics=[Metric("d_model", float(len(v) if isinstance(v, list) else 0))],
    )


@register(
    "merge_chatbot_roundtrip",
    suites=("smoke", "lm", "io"),
    description="merge(Chatbot, Chatbot) preserves shape",
)
def merge_chatbot_roundtrip(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del work_dir
    t0 = time.perf_counter()
    a = tiny_chatbot(seed)
    b = tiny_chatbot(seed + 1)
    merged = ai.merge(a, b)
    ok = merged.d_model == a.d_model and merged.vocab_size == a.vocab_size
    return TaskResult(
        name="merge_chatbot_roundtrip",
        ok=ok,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metrics=[Metric("parameters", float(merged.parameters))],
        meta={"d_model": merged.d_model, "vocab_size": merged.vocab_size},
    )


@register(
    "quantize_int8_roundtrip",
    suites=("lm", "io"),
    description="In-place int8 quantize keeps model usable",
)
def quantize_int8_roundtrip(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    work = ensure_work(work_dir)
    t0 = time.perf_counter()
    bot = tiny_chatbot(seed, n_layer=1)
    before = float(bot.compute_mean_loss(qa_dataset()))
    ai.quantize(bot, "int8")
    path = work / "q_int8.mmn"
    bot.save(str(path))
    loaded = ai.load(str(path))
    after = float(loaded.compute_mean_loss(qa_dataset()))
    ok = after == after and loaded.d_model == bot.d_model
    return TaskResult(
        name="quantize_int8_roundtrip",
        ok=ok,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metrics=[
            Metric("loss_before_quant", before, higher_is_better=False),
            Metric("loss_after_quant", after, higher_is_better=False),
        ],
    )
