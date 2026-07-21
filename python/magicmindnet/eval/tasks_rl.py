"""RL / SPIN smoke eval tasks."""

from __future__ import annotations

import time
from pathlib import Path

import magicmindnet as ai
from magicmindnet.eval._util import qa_dataset, tiny_chatbot, train_cfg
from magicmindnet.eval.registry import register
from magicmindnet.eval.types import Metric, TaskResult


@register("rl_policy_smoke", suites=("rl", "train"), description="RL policy mode one-step smoke")
def rl_policy_smoke(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del work_dir
    t0 = time.perf_counter()
    bot = tiny_chatbot(seed, n_layer=1)
    ds = qa_dataset()
    before = float(bot.compute_mean_loss(ds))
    cfg = train_cfg(epochs=1, lr=0.05)
    ai.RL(bot, ds, cfg, reward_amount=1.0, punishment_amount=0.5, rl_type="policy")
    after = float(bot.compute_mean_loss(ds))
    return TaskResult(
        name="rl_policy_smoke",
        ok=after == after,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metrics=[
            Metric("loss_before", before, higher_is_better=False),
            Metric("loss_after", after, higher_is_better=False),
        ],
        meta={"rl_type": "policy"},
    )


@register("spin_smoke", suites=("rl", "train"), description="SPIN self-play one epoch smoke")
def spin_smoke(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del work_dir
    t0 = time.perf_counter()
    bot = tiny_chatbot(seed + 3, n_layer=1)
    ds = qa_dataset()
    before = float(bot.compute_mean_loss(ds))
    ai.SPIN(bot, 1, ds)
    after = float(bot.compute_mean_loss(ds))
    return TaskResult(
        name="spin_smoke",
        ok=after == after,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metrics=[
            Metric("loss_before", before, higher_is_better=False),
            Metric("loss_after", after, higher_is_better=False),
        ],
    )
