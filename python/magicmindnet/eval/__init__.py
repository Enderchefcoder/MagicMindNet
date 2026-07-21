"""MagicMindNet eval harness — unified offline benchmarks across features.

Run a suite::

    import magicmindnet as ai

    report = ai.run_suite("smoke")
    print("\\n".join(report.summary_lines()))

CLI::

    python -m magicmindnet.eval smoke
    python -m magicmindnet.eval all --json -o report.json
"""

from __future__ import annotations

# Side-effect: register all tasks.
from magicmindnet.eval import (  # noqa: F401
    tasks_cls,
    tasks_diffusion,
    tasks_extra,
    tasks_hub,
    tasks_io,
    tasks_lm,
    tasks_rl,
)
from magicmindnet.eval.registry import get_task, list_suites, list_tasks
from magicmindnet.eval.runner import BenchmarkRunner, EvalHarness, run_suite
from magicmindnet.eval.types import Metric, SuiteReport, TaskResult

__all__ = [
    "BenchmarkRunner",
    "EvalHarness",
    "Metric",
    "SuiteReport",
    "TaskResult",
    "get_task",
    "list_suites",
    "list_tasks",
    "run_suite",
]
