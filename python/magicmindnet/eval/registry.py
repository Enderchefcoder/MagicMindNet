"""Task registration and suite membership for the eval harness."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from magicmindnet.eval.types import TaskResult

TaskFn = Callable[..., TaskResult]

_TASKS: dict[str, dict[str, Any]] = {}

# Canonical suite names → default task membership (filled by register + SUITE_EXTRA).
SUITE_ALIASES: dict[str, str] = {
    "classifier": "cls",
    "default": "smoke",
}

# Suites that are pure aliases of tag unions (no explicit list needed beyond tags).
TAG_SUITES: tuple[str, ...] = (
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
)


def register(
    name: str,
    *,
    suites: Sequence[str],
    description: str = "",
) -> Callable[[TaskFn], TaskFn]:
    """Decorator that registers a task function under ``name``."""

    def deco(fn: TaskFn) -> TaskFn:
        if name in _TASKS:
            raise ValueError(f"duplicate eval task: {name}")
        _TASKS[name] = {
            "name": name,
            "suites": tuple(suites),
            "description": description,
            "run": fn,
        }
        return fn

    return deco


def _normalize_suite(suite: str) -> str:
    return SUITE_ALIASES.get(suite, suite)


def list_suites() -> list[str]:
    return list(TAG_SUITES)


def list_tasks(suite: str | None = None) -> list[dict[str, Any]]:
    if suite is None:
        return [dict(t) for t in _TASKS.values()]
    suite = _normalize_suite(suite)
    if suite not in TAG_SUITES:
        raise KeyError(f"unknown eval suite: {suite!r}; known={list_suites()}")
    out: list[dict[str, Any]] = []
    for t in _TASKS.values():
        tags = set(t["suites"])
        if suite == "all" or suite in tags:
            out.append(dict(t))
    return out


def get_task(name: str) -> dict[str, Any]:
    try:
        return dict(_TASKS[name])
    except KeyError as exc:
        raise KeyError(f"unknown eval task: {name!r}") from exc


def resolve_task_names(
    suite: str | None = None,
    tasks: Iterable[str] | None = None,
) -> list[str]:
    if tasks is not None:
        names = list(tasks)
        for n in names:
            if n not in _TASKS:
                raise KeyError(f"unknown eval task: {n!r}")
        return names
    if suite is None:
        suite = "smoke"
    return [t["name"] for t in list_tasks(suite)]
