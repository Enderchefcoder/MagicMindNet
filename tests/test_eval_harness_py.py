"""Eval harness public API + offline suite regressions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import magicmindnet as ai


def test_eval_public_exports():
    assert hasattr(ai, "eval") or "EvalHarness" in dir(ai)
    from magicmindnet import eval as ev

    assert callable(ev.list_tasks)
    assert callable(ev.list_suites)
    assert callable(ev.run_suite)
    assert callable(ev.get_task)
    assert "EvalHarness" in ev.__all__
    assert "BenchmarkRunner" in ev.__all__
    assert "Metric" in ev.__all__
    assert "TaskResult" in ev.__all__
    assert "SuiteReport" in ev.__all__


def test_eval_exports_on_package_all():
    for name in (
        "EvalHarness",
        "BenchmarkRunner",
        "list_tasks",
        "list_suites",
        "run_suite",
        "get_task",
    ):
        assert name in ai.__all__
        assert getattr(ai, name) is not None


def test_list_suites_includes_canonical():
    from magicmindnet.eval import list_suites

    suites = list_suites()
    for name in (
        "smoke",
        "lm",
        "cls",
        "diffusion",
        "io",
        "hub",
        "glint",
        "generate",
        "rl",
        "train",
        "all",
    ):
        assert name in suites


def test_list_tasks_covers_feature_surface():
    from magicmindnet.eval import list_tasks

    names = {t["name"] for t in list_tasks()}
    required = {
        "lm_qa_loss",
        "lm_qa_train",
        "lm_corpus_loss",
        "lm_corpus_train",
        "lm_learned_pe_train",
        "lm_rope_train",
        "lm_bpe_train",
        "lm_gqa_train",
        "lm_glint_train",
        "lm_glint2_exact",
        "lm_generate_latency",
        "lm_head_dim_train",
        "lm_unigram_train",
        "lm_vision_flag",
        "cls_mean_loss",
        "cls_train",
        "cls_accuracy",
        "diffusion_denoise",
        "diffusion_edit_denoise",
        "diffusion_train",
        "diffusion_edit_train",
        "io_roundtrip_safetensors",
        "io_roundtrip_gguf",
        "io_roundtrip_npz",
        "io_roundtrip_pt",
        "io_roundtrip_hf_safetensors",
        "io_roundtrip_gguf_q8",
        "arrays_npz_roundtrip",
        "hub_causal_generate",
        "hub_classifier_predict",
        "hub_rerank",
        "hub_diffusion_generate",
        "hub_seq2seq_generate",
        "hub_capabilities_matrix",
        "rl_policy_smoke",
        "spin_smoke",
        "merge_chatbot_roundtrip",
        "quantize_int8_roundtrip",
    }
    missing = required - names
    assert not missing, f"missing tasks: {sorted(missing)}"


def test_smoke_suite_runs_and_passes(tmp_path: Path):
    from magicmindnet.eval import EvalHarness, run_suite

    report = run_suite("smoke", seed=7, work_dir=tmp_path)
    assert report.suite == "smoke"
    assert report.ok
    assert report.n_tasks >= 5
    assert all(r.ok for r in report.results)
    d = report.to_dict()
    assert d["ok"] is True
    assert "elapsed_ms" in d
    path = tmp_path / "report.json"
    report.write_json(path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["suite"] == "smoke"
    assert "smoke" in EvalHarness.list_suites()


def test_get_task_and_run_single(tmp_path: Path):
    from magicmindnet.eval import get_task

    task = get_task("lm_qa_loss")
    assert task["name"] == "lm_qa_loss"
    assert "lm" in task["suites"]
    result = task["run"](seed=3, work_dir=tmp_path)
    assert result.ok
    assert result.name == "lm_qa_loss"
    metrics = {m.name: m.value for m in result.metrics}
    assert "mean_loss" in metrics
    assert metrics["mean_loss"] > 0


def test_lm_qa_train_reduces_loss(tmp_path: Path):
    from magicmindnet.eval import get_task

    result = get_task("lm_qa_train")["run"](seed=11, work_dir=tmp_path)
    assert result.ok
    metrics = {m.name: m.value for m in result.metrics}
    assert metrics["loss_after"] < metrics["loss_before"]
    assert metrics["loss_delta"] < 0


def test_cls_train_and_accuracy(tmp_path: Path):
    from magicmindnet.eval import get_task

    train = get_task("cls_train")["run"](seed=5, work_dir=tmp_path)
    assert train.ok
    acc = get_task("cls_accuracy")["run"](seed=5, work_dir=tmp_path)
    assert acc.ok
    metrics = {m.name: m.value for m in acc.metrics}
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_glint_and_gqa_tasks(tmp_path: Path):
    from magicmindnet.eval import get_task

    for name in ("lm_glint_train", "lm_gqa_train"):
        result = get_task(name)["run"](seed=9, work_dir=tmp_path)
        assert result.ok, result.error
        metrics = {m.name: m.value for m in result.metrics}
        assert metrics["loss_after"] <= metrics["loss_before"]


def test_io_roundtrip_tasks(tmp_path: Path):
    from magicmindnet.eval import get_task

    for name in (
        "io_roundtrip_safetensors",
        "io_roundtrip_gguf",
        "io_roundtrip_npz",
        "io_roundtrip_pt",
        "io_roundtrip_hf_safetensors",
    ):
        result = get_task(name)["run"](seed=2, work_dir=tmp_path)
        assert result.ok, f"{name}: {result.error}"
        metrics = {m.name: m.value for m in result.metrics}
        assert metrics["size_bytes"] > 0
        assert metrics["save_ms"] >= 0
        assert metrics["load_ms"] >= 0


def test_hub_synthetic_tasks(tmp_path: Path):
    from magicmindnet.eval import get_task

    for name in (
        "hub_causal_generate",
        "hub_classifier_predict",
        "hub_rerank",
        "hub_diffusion_generate",
        "hub_seq2seq_generate",
        "hub_capabilities_matrix",
    ):
        result = get_task(name)["run"](seed=1, work_dir=tmp_path)
        assert result.ok, f"{name}: {result.error}"


def test_diffusion_and_rl_tasks(tmp_path: Path):
    from magicmindnet.eval import get_task

    for name in (
        "diffusion_denoise",
        "diffusion_edit_denoise",
        "rl_policy_smoke",
        "spin_smoke",
        "merge_chatbot_roundtrip",
        "quantize_int8_roundtrip",
    ):
        result = get_task(name)["run"](seed=4, work_dir=tmp_path)
        assert result.ok, f"{name}: {result.error}"


def test_eval_harness_filter_and_fail_fast(tmp_path: Path):
    from magicmindnet.eval import EvalHarness

    harness = EvalHarness(seed=1, work_dir=tmp_path)
    report = harness.run(tasks=["lm_qa_loss", "cls_mean_loss"])
    assert report.ok
    assert [r.name for r in report.results] == ["lm_qa_loss", "cls_mean_loss"]


def test_unknown_suite_and_task_raise():
    from magicmindnet.eval import get_task, run_suite

    with pytest.raises(KeyError):
        get_task("not_a_real_task")
    with pytest.raises(KeyError):
        run_suite("not_a_real_suite")


def test_extra_feature_tasks(tmp_path: Path):
    from magicmindnet.eval import get_task

    for name in (
        "lm_head_dim_train",
        "lm_unigram_train",
        "lm_vision_flag",
        "io_roundtrip_gguf_q8",
        "arrays_npz_roundtrip",
        "diffusion_edit_train",
    ):
        result = get_task(name)["run"](seed=6, work_dir=tmp_path)
        assert result.ok, f"{name}: {result.error}"


def test_all_suite_completes(tmp_path: Path):
    from magicmindnet.eval import run_suite

    report = run_suite("all", seed=2, work_dir=tmp_path)
    assert report.n_tasks >= 30
    failed = [r.name for r in report.results if not r.ok]
    assert not failed, f"failed tasks: {failed}"


def test_eval_harness_example_script(run_example):
    result = run_example("eval_harness.py", "smoke", "--json")
    assert result.returncode == 0, result.stderr + result.stdout
    assert "smoke" in result.stdout.lower() or '"ok"' in result.stdout
