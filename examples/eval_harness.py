"""Thin CLI demo for the unified eval harness.

Examples::

    python examples/eval_harness.py smoke
    python examples/eval_harness.py lm --json
    python examples/eval_harness.py --list-tasks
    python -m magicmindnet.eval all -o /tmp/mmn_eval.json
"""

from __future__ import annotations

import sys

from magicmindnet.eval.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
