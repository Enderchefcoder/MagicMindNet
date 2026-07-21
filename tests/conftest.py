"""Shared pytest helpers for subprocess example smoke tests and checkpoint introspection."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def tensor_entry_first_f32(tensor_entry: dict) -> float:
    """Decode the first little-endian f32 in a checkpoint tensor entry dict."""
    data = tensor_entry["data"]
    return struct.unpack("<f", bytes(data[:4]))[0]


def tamper_tensor_entry_first_f32(tensor_entry: dict, value: float) -> None:
    """Overwrite the first f32 in-place inside a tensor entry's `data` list."""
    data = tensor_entry["data"]
    tensor_entry["data"] = list(struct.pack("<f", value) + bytes(data[4:]))


def load_checkpoint(export_path: Path) -> dict:
    """Load full mmn JSON checkpoint (`meta` + `tensors`)."""
    return json.loads(export_path.read_text(encoding="utf-8"))


def load_checkpoint_tensors(export_path: Path) -> dict:
    """Load the `tensors` map from an mmn-safetensors-v1 JSON export."""
    return load_checkpoint(export_path)["tensors"]


def checkpoint_tensor_bytes(export_path: Path, key: str) -> list:
    """Return raw tensor `data` bytes as a list (for equality checks after train/export)."""
    return list(load_checkpoint_tensors(export_path)[key]["data"])


def checkpoint_tensor_first_f32(export_path: Path, key: str) -> float:
    """Decode the first little-endian f32 in a checkpoint tensor entry."""
    return tensor_entry_first_f32(load_checkpoint_tensors(export_path)[key])


def project_python() -> str:
    venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.is_file():
        return str(venv_py)
    venv_py_unix = ROOT / ".venv" / "bin" / "python"
    if venv_py_unix.is_file():
        return str(venv_py_unix)
    return sys.executable


def install_fake_torch():
    """Register stand-in torch modules so CPython pickle can resolve globals.

    Idempotent: repeated calls return the same module objects so classes
    pickled by reference stay identical across tests.
    """
    import types

    if "torch" in sys.modules and getattr(sys.modules["torch"], "__file__", None):
        # Real torch already imported (optional hub deps). Do not replace it —
        # callers that need stub rebuild (pt pickle cross-checks) must supply
        # their own find_class stubs. Returning the live modules keeps Trainer OK.
        return sys.modules["torch"], sys.modules.get("torch._utils") or sys.modules["torch"]
    if "torch" in sys.modules:
        return sys.modules["torch"], sys.modules["torch._utils"]
    torch = types.ModuleType("torch")
    torch_utils = types.ModuleType("torch._utils")

    class FloatStorage:
        pass

    def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad, hooks):
        return {"storage": storage, "offset": offset, "size": size, "stride": stride}

    FloatStorage.__module__ = "torch"
    FloatStorage.__qualname__ = "FloatStorage"
    _rebuild_tensor_v2.__module__ = "torch._utils"
    _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
    torch.FloatStorage = FloatStorage
    torch._utils = torch_utils
    torch_utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    sys.modules["torch"] = torch
    sys.modules["torch._utils"] = torch_utils
    return torch, torch_utils


def run_example_script(
    script: str,
    *script_args: str,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [project_python(), str(ROOT / "examples" / script), *script_args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@pytest.fixture(scope="session")
def project_python_exe() -> str:
    return project_python()


@pytest.fixture
def run_example(project_python_exe: str):
    def _run(
        script: str,
        *script_args: str,
        timeout: int = 120,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [project_python_exe, str(ROOT / "examples" / script), *script_args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    return _run
