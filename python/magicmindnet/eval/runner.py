"""EvalHarness / BenchmarkRunner — run registered suites and tasks."""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

from magicmindnet.eval.registry import (
    get_task,
    list_suites,
    list_tasks,
    resolve_task_names,
)
from magicmindnet.eval.types import SuiteReport, TaskResult


class EvalHarness:
    """Run offline eval / benchmark tasks against MagicMindNet features."""

    def __init__(
        self,
        *,
        seed: int = 1,
        work_dir: str | Path | None = None,
        fail_fast: bool = False,
        verbose: bool = False,
    ) -> None:
        self.seed = seed
        self.work_dir = Path(work_dir) if work_dir is not None else None
        self.fail_fast = fail_fast
        self.verbose = verbose

    @staticmethod
    def list_suites() -> list[str]:
        return list_suites()

    @staticmethod
    def list_tasks(suite: str | None = None) -> list[dict[str, Any]]:
        return list_tasks(suite)

    @staticmethod
    def get_task(name: str) -> dict[str, Any]:
        return get_task(name)

    def run(
        self,
        *,
        suite: str | None = None,
        tasks: list[str] | None = None,
    ) -> SuiteReport:
        names = resolve_task_names(suite=suite, tasks=tasks)
        label = suite or ("custom" if tasks else "smoke")
        t0 = time.perf_counter()
        results: list[TaskResult] = []
        work = self.work_dir
        if work is not None:
            work.mkdir(parents=True, exist_ok=True)

        for name in names:
            task = get_task(name)
            if self.verbose:
                print(f"→ {name} …", flush=True)
            started = time.perf_counter()
            try:
                kwargs: dict[str, Any] = {"seed": self.seed}
                if work is not None:
                    kwargs["work_dir"] = work
                result = task["run"](**kwargs)
                if not isinstance(result, TaskResult):
                    raise TypeError(f"task {name} returned {type(result)!r}, expected TaskResult")
                if result.elapsed_ms <= 0:
                    result.elapsed_ms = (time.perf_counter() - started) * 1000
            except Exception as exc:  # noqa: BLE001 — harness must capture task failures
                result = TaskResult(
                    name=name,
                    ok=False,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    error=f"{type(exc).__name__}: {exc}",
                    meta={"traceback": traceback.format_exc()},
                )
            results.append(result)
            if self.verbose:
                status = "ok" if result.ok else "FAIL"
                print(f"  {status} ({result.elapsed_ms:.1f} ms)", flush=True)
            if self.fail_fast and not result.ok:
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000
        ok = all(r.ok for r in results) and bool(results)
        return SuiteReport(
            suite=label,
            ok=ok,
            results=results,
            elapsed_ms=elapsed_ms,
            seed=self.seed,
            meta={"fail_fast": self.fail_fast},
        )

    def run_suite(self, suite: str) -> SuiteReport:
        return self.run(suite=suite)


# Friendly alias used in docs / exports.
BenchmarkRunner = EvalHarness


def run_suite(
    suite: str = "smoke",
    *,
    seed: int = 1,
    work_dir: str | Path | None = None,
    fail_fast: bool = False,
    verbose: bool = False,
) -> SuiteReport:
    """Run a named suite and return a :class:`SuiteReport`."""
    return EvalHarness(
        seed=seed,
        work_dir=work_dir,
        fail_fast=fail_fast,
        verbose=verbose,
    ).run_suite(suite)
