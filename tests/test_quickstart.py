"""Smoke-test examples/quickstart.py."""

from conftest import run_example_script


def test_quickstart_example_runs():
    proc = run_example_script("quickstart.py", timeout=60)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "Training finished" in proc.stdout
