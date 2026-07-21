"""CLI: ``python -m magicmindnet.eval [suite|task...] [options]``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from magicmindnet.eval import EvalHarness, list_suites, list_tasks, run_suite


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m magicmindnet.eval",
        description="MagicMindNet unified eval / benchmark harness",
    )
    p.add_argument(
        "targets",
        nargs="*",
        default=["smoke"],
        help="Suite name (smoke|lm|cls|…) and/or task names. Default: smoke",
    )
    p.add_argument("--list-suites", action="store_true", help="Print suite names and exit")
    p.add_argument(
        "--list-tasks",
        nargs="?",
        const="all",
        default=None,
        metavar="SUITE",
        help="List tasks (optionally filtered by suite) and exit",
    )
    p.add_argument("--seed", type=int, default=1, help="RNG seed for deterministic tasks")
    p.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Scratch directory for IO / classification fixtures",
    )
    p.add_argument("--fail-fast", action="store_true", help="Stop on first failing task")
    p.add_argument("--verbose", "-v", action="store_true", help="Print per-task progress")
    p.add_argument("--json", action="store_true", help="Emit JSON report to stdout")
    p.add_argument("-o", "--output", type=Path, default=None, help="Write JSON report to path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.list_suites:
        print("\n".join(list_suites()))
        return 0

    if args.list_tasks is not None:
        suite = None if args.list_tasks == "all" else args.list_tasks
        for t in list_tasks(suite):
            tags = ",".join(t["suites"])
            desc = t.get("description") or ""
            print(f"{t['name']:<32} [{tags}] {desc}")
        return 0

    suites = set(list_suites())
    suite_targets = [t for t in args.targets if t in suites]
    task_targets = [t for t in args.targets if t not in suites]

    if suite_targets and task_targets:
        print("Pass either suite names or task names, not both.", file=sys.stderr)
        return 2
    if len(suite_targets) > 1:
        print("Pass a single suite name (or a list of task names).", file=sys.stderr)
        return 2

    harness = EvalHarness(
        seed=args.seed,
        work_dir=args.work_dir,
        fail_fast=args.fail_fast,
        verbose=args.verbose,
    )
    if task_targets:
        report = harness.run(tasks=task_targets)
    else:
        suite = suite_targets[0] if suite_targets else "smoke"
        report = run_suite(
            suite,
            seed=args.seed,
            work_dir=args.work_dir,
            fail_fast=args.fail_fast,
            verbose=args.verbose,
        )

    if args.output is not None:
        report.write_json(args.output)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print("\n".join(report.summary_lines()))

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
