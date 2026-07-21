"""Checkpoint IO roundtrip timing / size eval tasks."""

from __future__ import annotations

import os
import time
from pathlib import Path

import magicmindnet as ai
from magicmindnet.eval._util import ensure_work, tiny_chatbot
from magicmindnet.eval.registry import register
from magicmindnet.eval.types import Metric, TaskResult

_FORMATS = (
    ("io_roundtrip_safetensors", "safetensors", "rt.mmn", ("smoke", "io")),
    ("io_roundtrip_gguf", "gguf", "rt.gguf", ("io",)),
    ("io_roundtrip_npz", "npz", "rt.npz", ("io",)),
    ("io_roundtrip_pt", "pt", "rt.pt", ("io",)),
    ("io_roundtrip_hf_safetensors", "hf-safetensors", "rt.safetensors", ("io",)),
)


def _make_io_task(name: str, fmt: str, filename: str):
    def _run(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
        work = ensure_work(work_dir)
        path = work / filename
        bot = tiny_chatbot(seed, n_layer=1, d_model=64, vocab_size=128)
        t0 = time.perf_counter()
        bot.save(str(path), format=fmt)
        save_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        loaded = ai.load(str(path))
        load_ms = (time.perf_counter() - t1) * 1000
        size = float(os.path.getsize(path))
        ok = loaded.d_model == bot.d_model and loaded.vocab_size == bot.vocab_size
        path.unlink(missing_ok=True)
        return TaskResult(
            name=name,
            ok=ok,
            elapsed_ms=save_ms + load_ms,
            metrics=[
                Metric("size_bytes", size, unit="B", higher_is_better=False),
                Metric("save_ms", save_ms, unit="ms", higher_is_better=False),
                Metric("load_ms", load_ms, unit="ms", higher_is_better=False),
            ],
            meta={"format": fmt},
        )

    return _run


for _name, _fmt, _file, _suites in _FORMATS:
    register(_name, suites=_suites, description=f"Save/load timing for format={_fmt}")(
        _make_io_task(_name, _fmt, _file)
    )
