"""Shared types for the MagicMindNet eval harness."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Metric:
    """A single named measurement from a benchmark task."""

    name: str
    value: float
    unit: str = ""
    higher_is_better: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskResult:
    """Outcome of one registered eval task."""

    name: str
    ok: bool
    metrics: list[Metric] = field(default_factory=list)
    elapsed_ms: float = 0.0
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
            "meta": self.meta,
            "metrics": [m.to_dict() for m in self.metrics],
        }


@dataclass
class SuiteReport:
    """Aggregated results for a named suite or custom task list."""

    suite: str
    ok: bool
    results: list[TaskResult] = field(default_factory=list)
    elapsed_ms: float = 0.0
    seed: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_tasks(self) -> int:
        return len(self.results)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.ok)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "ok": self.ok,
            "elapsed_ms": self.elapsed_ms,
            "seed": self.seed,
            "n_tasks": self.n_tasks,
            "n_failed": self.n_failed,
            "meta": self.meta,
            "results": [r.to_dict() for r in self.results],
        }

    def write_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def summary_lines(self) -> list[str]:
        status = "PASS" if self.ok else "FAIL"
        lines = [
            f"suite={self.suite} status={status} tasks={self.n_tasks} "
            f"failed={self.n_failed} elapsed_ms={self.elapsed_ms:.1f}"
        ]
        for r in self.results:
            mark = "ok" if r.ok else "FAIL"
            metric_bits = ", ".join(
                f"{m.name}={m.value:.4g}{(' ' + m.unit) if m.unit else ''}"
                for m in r.metrics[:6]
            )
            extra = f" [{metric_bits}]" if metric_bits else ""
            err = f" err={r.error}" if r.error else ""
            lines.append(f"  {mark} {r.name} ({r.elapsed_ms:.1f} ms){extra}{err}")
        return lines
