"""Diffusion denoise / train eval tasks."""

from __future__ import annotations

import time
from pathlib import Path

import magicmindnet as ai
from magicmindnet.eval._util import fixtures_dir
from magicmindnet.eval.registry import register
from magicmindnet.eval.types import Metric, TaskResult


@register(
    "diffusion_denoise",
    suites=("smoke", "diffusion"),
    description="Mean denoise loss on image_gen fixture",
)
def diffusion_denoise(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del seed, work_dir
    t0 = time.perf_counter()
    ds = ai.DatasetImageGen(file=str(fixtures_dir() / "image_gen.json"))
    d = ai.Diffusion()
    loss = float(d.compute_mean_denoise_loss(ds, t=7))
    return TaskResult(
        name="diffusion_denoise",
        ok=loss == loss,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metrics=[Metric("mean_denoise_loss", loss, higher_is_better=False)],
    )


@register(
    "diffusion_edit_denoise",
    suites=("diffusion",),
    description="Mean denoise loss on image_edit (inpaint) fixture",
)
def diffusion_edit_denoise(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del seed, work_dir
    t0 = time.perf_counter()
    ds = ai.DatasetImageEdit(file=str(fixtures_dir() / "image_edit.json"))
    d = ai.Diffusion()
    loss = float(d.compute_mean_denoise_loss(ds, t=5))
    return TaskResult(
        name="diffusion_edit_denoise",
        ok=loss == loss,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metrics=[Metric("mean_denoise_loss", loss, higher_is_better=False)],
    )


@register(
    "diffusion_train",
    suites=("diffusion", "train"),
    description="TrainDiffusion mean denoise loss delta",
)
def diffusion_train(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del seed, work_dir
    t0 = time.perf_counter()
    ds = ai.DatasetImageGen(file=str(fixtures_dir() / "image_gen.json"))
    d = ai.Diffusion()
    before = float(d.compute_mean_denoise_loss(ds, t=7))
    cfg = ai.TrainConfig(epochs=6, batch_size=1, learning_rate=0.05, cuda=False)
    ai.TrainDiffusion(d, ds, cfg)
    after = float(d.compute_mean_denoise_loss(ds, t=7))
    return TaskResult(
        name="diffusion_train",
        ok=after == after,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metrics=[
            Metric("loss_before", before, higher_is_better=False),
            Metric("loss_after", after, higher_is_better=False),
            Metric("loss_delta", after - before, higher_is_better=False),
        ],
    )
