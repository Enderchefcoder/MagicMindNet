"""Hub synthetic-family eval tasks (offline, no network)."""

from __future__ import annotations

import time
from pathlib import Path

import magicmindnet as ai
from magicmindnet.eval.registry import register
from magicmindnet.eval.types import Metric, TaskResult
from magicmindnet.hub import HubModel, list_hub_families


@register(
    "hub_causal_generate",
    suites=("smoke", "hub", "generate"),
    description="Synthetic HubModel causal-lm generate",
)
def hub_causal_generate(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del seed, work_dir
    t0 = time.perf_counter()
    hub = HubModel.synthetic_causal_lm()
    text = hub.generate("hi", max_new_tokens=4)
    caps = hub.capabilities()
    ok = caps.get("generate", False) and isinstance(text, str)
    return TaskResult(
        name="hub_causal_generate",
        ok=ok,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metrics=[Metric("max_new_tokens", 4.0)],
        meta={"family": hub.family, "preview": str(text)[:64]},
    )


@register(
    "hub_classifier_predict",
    suites=("hub", "cls"),
    description="Synthetic HubModel classifier predict_label",
)
def hub_classifier_predict(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del seed, work_dir
    t0 = time.perf_counter()
    hub = HubModel.synthetic_classifier(["pos", "neg"])
    label = hub.predict_label("hello")
    probs = hub.predict("hello")
    ok = label in ("pos", "neg") and probs is not None
    return TaskResult(
        name="hub_classifier_predict",
        ok=ok,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metrics=[Metric("n_labels", 2.0)],
        meta={"label": str(label)},
    )


@register("hub_rerank", suites=("hub",), description="Synthetic reranker score_pairs/rerank")
def hub_rerank(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del seed, work_dir
    t0 = time.perf_counter()
    hub = HubModel.synthetic_reranker()
    docs = ["cats and dogs", "quantum physics", "cat food"]
    scores = hub.score_pairs("cat", docs)
    ranked = hub.rerank("cat", docs)
    ok = len(scores) == 3 and len(ranked) == 3 and scores[0] >= 0
    return TaskResult(
        name="hub_rerank",
        ok=ok,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metrics=[
            Metric("top_score", float(max(scores))),
            Metric("n_docs", float(len(docs))),
        ],
    )


@register(
    "hub_diffusion_generate",
    suites=("hub", "diffusion", "generate"),
    description="Synthetic HubModel diffusion generate",
)
def hub_diffusion_generate(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del seed, work_dir
    t0 = time.perf_counter()
    hub = HubModel.synthetic_diffusion()
    out = hub.generate("a cat", steps=2)
    ok = out is not None and hub.capabilities().get("generate", False)
    return TaskResult(
        name="hub_diffusion_generate",
        ok=ok,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metrics=[Metric("steps", 2.0)],
        meta={"type": type(out).__name__},
    )


@register(
    "hub_seq2seq_generate",
    suites=("hub", "generate"),
    description="Synthetic HubModel seq2seq generate",
)
def hub_seq2seq_generate(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del seed, work_dir
    t0 = time.perf_counter()
    hub = HubModel.synthetic_seq2seq()
    text = hub.generate("hello", max_new_tokens=3)
    ok = isinstance(text, str) and hub.capabilities().get("generate", False)
    return TaskResult(
        name="hub_seq2seq_generate",
        ok=ok,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metrics=[Metric("max_new_tokens", 3.0)],
        meta={"preview": str(text)[:64]},
    )


@register(
    "hub_capabilities_matrix",
    suites=("smoke", "hub"),
    description="list_hub_families + synthetic capability flags",
)
def hub_capabilities_matrix(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    del seed, work_dir
    t0 = time.perf_counter()
    families = list_hub_families()
    required = {
        "causal-lm",
        "classifier",
        "reranker",
        "seq2seq",
        "diffusion",
        "embedding",
        "gguf",
    }
    missing = required - set(families)
    samples = [
        HubModel.synthetic_causal_lm(),
        HubModel.synthetic_classifier(["a", "b"]),
        HubModel.synthetic_diffusion(),
        HubModel.synthetic_reranker(),
        HubModel.synthetic_seq2seq(),
    ]
    cap_ok = all(isinstance(h.capabilities(), dict) for h in samples)
    ok = not missing and cap_ok and "from_pretrained" in ai.__all__
    return TaskResult(
        name="hub_capabilities_matrix",
        ok=ok,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metrics=[
            Metric("n_families", float(len(families))),
            Metric("n_synthetics", float(len(samples))),
        ],
        meta={"missing_families": sorted(missing)},
    )
