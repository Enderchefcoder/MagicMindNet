"""Regression tests for shared helpers in tests/conftest.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import (
    ROOT,
    project_python,
    tamper_tensor_entry_first_f32,
    tensor_entry_first_f32,
)


def test_tensor_entry_first_f32_decodes_le_float():
    entry = {"data": list(b"\x00\x00\x80\x3f") + [0] * 4, "shape": [2]}
    assert tensor_entry_first_f32(entry) == pytest.approx(1.0)


def test_tamper_tensor_entry_first_f32_roundtrip():
    entry = {"data": [0] * 8, "shape": [2]}
    tamper_tensor_entry_first_f32(entry, 2.5)
    assert tensor_entry_first_f32(entry) == pytest.approx(2.5)


def test_load_checkpoint_tensors_matches_tensors_key(tmp_path: Path):
    from conftest import load_checkpoint, load_checkpoint_tensors

    path = tmp_path / "fake.mmn"
    payload = {
        "meta": {"vocab_size": 64},
        "tensors": {"embed": {"data": [0, 0, 128, 63], "shape": [1, 1]}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_checkpoint_tensors(path) == payload["tensors"]
    assert load_checkpoint(path)["meta"]["vocab_size"] == 64


@pytest.mark.skipif(not (ROOT / ".venv").exists(), reason="no .venv")
def test_project_python_matches_venv_resolver_script():
    resolved = Path(project_python()).resolve()
    if sys.platform == "win32":
        script = ROOT / "scripts" / "venv_python.ps1"
        expected = subprocess.check_output(
            ["powershell", "-NoProfile", "-File", str(script)],
            text=True,
        ).strip()
    else:
        script = ROOT / "scripts" / "venv_python.sh"
        expected = subprocess.check_output(["bash", str(script)], text=True).strip()
    assert resolved == Path(expected).resolve()
