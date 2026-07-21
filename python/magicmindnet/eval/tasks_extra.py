"""Extra feature-surface eval tasks: head_dim, vision, unigram, arrays, quant formats."""

from __future__ import annotations

import os
import time
from pathlib import Path

import magicmindnet as ai
from magicmindnet.eval._util import ensure_work, fixtures_dir, qa_dataset, tiny_chatbot, train_cfg
from magicmindnet.eval.registry import register
from magicmindnet.eval.types import Metric, TaskResult
from magicmindnet.interop import load_arrays, save_arrays


@register(
    "lm_head_dim_train",
    suites=("lm", "train"),
    description="Optional head_dim (Qwen-style) train delta",
)
def lm_head_dim_train(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del work_dir
    t0 = time.perf_counter()
    bot = tiny_chatbot(seed, d_model=64, n_heads=4, head_dim=32, n_layer=1)
    ds = qa_dataset()
    before = float(bot.compute_mean_loss(ds))
    ai.Train(bot, ds, train_cfg(epochs=3))
    after = float(bot.compute_mean_loss(ds))
    return TaskResult(
        name="lm_head_dim_train",
        ok=after <= before + 1e-6,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metrics=[
            Metric("loss_before", before, higher_is_better=False),
            Metric("loss_after", after, higher_is_better=False),
            Metric("loss_delta", after - before, higher_is_better=False),
        ],
        meta={"head_dim": bot.head_dim, "n_heads": bot.n_heads},
    )


@register(
    "lm_unigram_train",
    suites=("lm", "train"),
    description="Unigram-tokenized QA train delta",
)
def lm_unigram_train(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del work_dir
    t0 = time.perf_counter()
    vocab = 128
    bot = tiny_chatbot(seed, vocab_size=vocab, n_layer=1)
    enc = ai.UnigramEncoder.train(["hello world foo bar baz"] * 24, vocab_size=vocab)
    ds = qa_dataset()
    before = float(bot.compute_mean_loss(ds, unigram_encoder=enc))
    ai.Train(bot, ds, train_cfg(epochs=3), unigram_encoder=enc)
    after = float(bot.compute_mean_loss(ds, unigram_encoder=enc))
    return TaskResult(
        name="lm_unigram_train",
        ok=after <= before + 1e-6,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metrics=[
            Metric("loss_before", before, higher_is_better=False),
            Metric("loss_after", after, higher_is_better=False),
            Metric("loss_delta", after - before, higher_is_better=False),
        ],
        meta={"unigram_pieces": enc.piece_count},
    )


@register(
    "lm_vision_flag",
    suites=("lm", "smoke"),
    description="Vision-enabled Chatbot constructs and generates",
)
def lm_vision_flag(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del work_dir
    t0 = time.perf_counter()
    bot = ai.Chatbot(vocab_size=128, n_layer=2, d_model=32, seed=seed, vision=True)
    text = bot.generate("hi", max_new_tokens=2)
    ok = bool(bot.has_vision) and isinstance(text, str)
    return TaskResult(
        name="lm_vision_flag",
        ok=ok,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metrics=[
            Metric("vision_patch_dim", float(bot.vision_patch_dim)),
            Metric("has_vision", 1.0 if bot.has_vision else 0.0),
        ],
    )


@register(
    "io_roundtrip_gguf_q8",
    suites=("io",),
    description="GGUF Q8_0 save/load timing + size",
)
def io_roundtrip_gguf_q8(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    work = ensure_work(work_dir)
    path = work / "rt_q8.gguf"
    bot = tiny_chatbot(seed, n_layer=1, d_model=64, vocab_size=128)
    t0 = time.perf_counter()
    bot.save(str(path), format="gguf-q8_0")
    save_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    loaded = ai.load(str(path))
    load_ms = (time.perf_counter() - t1) * 1000
    size = float(os.path.getsize(path))
    ok = loaded.d_model == bot.d_model
    path.unlink(missing_ok=True)
    return TaskResult(
        name="io_roundtrip_gguf_q8",
        ok=ok,
        elapsed_ms=save_ms + load_ms,
        metrics=[
            Metric("size_bytes", size, unit="B", higher_is_better=False),
            Metric("save_ms", save_ms, unit="ms", higher_is_better=False),
            Metric("load_ms", load_ms, unit="ms", higher_is_better=False),
        ],
        meta={"format": "gguf-q8_0"},
    )


@register(
    "arrays_npz_roundtrip",
    suites=("io", "smoke"),
    description="ai.save_arrays / load_arrays NPZ roundtrip",
)
def arrays_npz_roundtrip(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del seed
    work = ensure_work(work_dir)
    path = work / "arrays.npz"
    t0 = time.perf_counter()
    payload = {"w": [[1.0, 2.0], [3.0, 4.0]], "b": [0.5, -0.25]}
    save_arrays(str(path), payload)
    loaded = load_arrays(str(path))
    ok = "w" in loaded and "b" in loaded
    size = float(os.path.getsize(path))
    path.unlink(missing_ok=True)
    return TaskResult(
        name="arrays_npz_roundtrip",
        ok=ok,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metrics=[
            Metric("size_bytes", size, unit="B", higher_is_better=False),
            Metric("n_tensors", float(len(loaded))),
        ],
    )


@register(
    "diffusion_edit_train",
    suites=("diffusion", "train"),
    description="TrainDiffusion on image_edit (inpaint) fixture",
)
def diffusion_edit_train(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del seed, work_dir
    t0 = time.perf_counter()
    ds = ai.DatasetImageEdit(file=str(fixtures_dir() / "image_edit.json"))
    d = ai.Diffusion()
    before = float(d.compute_mean_denoise_loss(ds, t=5))
    cfg = ai.TrainConfig(epochs=6, batch_size=1, learning_rate=0.05, cuda=False)
    ai.TrainDiffusion(d, ds, cfg)
    after = float(d.compute_mean_denoise_loss(ds, t=5))
    return TaskResult(
        name="diffusion_edit_train",
        ok=after == after,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metrics=[
            Metric("loss_before", before, higher_is_better=False),
            Metric("loss_after", after, higher_is_better=False),
            Metric("loss_delta", after - before, higher_is_better=False),
        ],
    )
