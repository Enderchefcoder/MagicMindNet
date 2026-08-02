"""DistribAI script jobs — packaging, preflight mirror, and the worker contract.

The packaging checks mirror ``services_python/preflight.py`` and
``services_python/script_validation.py``; the end-to-end test replays what
``worker/src/daemon/script_runner.py`` does on a node (unpack, write
``hyperparams.json``, set ``DISTRIBAI_*`` env, run ``run.py``, collect
``results.json`` / ``metrics.json`` / ``checkpoint.pt``).
"""

import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from conftest import project_python
from magicmindnet import distribai as bridge

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Static script validation (DistribAI's AST rules)
# ---------------------------------------------------------------------------


def test_default_run_py_passes_distribai_static_validation():
    assert bridge.validate_script_source(bridge.default_run_py()) == []


@pytest.mark.parametrize(
    "source,code",
    [
        ("", "empty_script"),
        ("def broken(:", "syntax_error"),
        ("eval('1+1')", "disallowed_call:eval"),
        ("exec('x=1')", "disallowed_call:exec"),
        ("import subprocess", "disallowed_import:subprocess"),
        ("from multiprocessing import Pool", "disallowed_import_from:multiprocessing"),
        ("import ctypes.util", "disallowed_import:ctypes.util"),
    ],
)
def test_validate_script_source_flags_violations(source, code):
    errors = bridge.validate_script_source(source)
    assert any(error.startswith(code) for error in errors), errors


# ---------------------------------------------------------------------------
# Package building + preflight mirror
# ---------------------------------------------------------------------------


def test_build_script_package_layout_and_determinism(tmp_path):
    config = {"job_type": "train", "architecture_config": bridge.GRID_PROFILES["mmn-tiny"]}
    dataset = [{"input": "hi", "output": "yo"}]
    package = bridge.build_script_package(config=config, dataset=dataset)
    again = bridge.build_script_package(config=config, dataset=dataset)
    assert package == again  # deterministic tar (sorted names, zeroed mtimes)

    with tarfile.open(fileobj=io.BytesIO(package), mode="r:gz") as tar:
        names = sorted(member.name for member in tar.getmembers())
        assert names == ["config.json", "dataset.json", "requirements.txt", "run.py"]
        requirements = tar.extractfile("requirements.txt").read().decode()
    assert "magicmindnet" in requirements

    out = tmp_path / "bundle" / "package.tar.gz"
    written = bridge.build_script_package(config=config, dataset=dataset, out=out)
    assert out.read_bytes() == written


def test_build_script_package_validates_and_reports_meta():
    package = bridge.build_script_package()
    meta = bridge.validate_script_package(package)
    assert meta["member_count"] >= 2
    assert meta["size_bytes"] == len(package)
    assert len(bridge.package_sha256(package)) == 64


def test_build_script_package_rejects_invalid_run_py():
    with pytest.raises(ValueError, match="failed validation.*disallowed_call:eval"):
        bridge.build_script_package("eval('1')")


def test_build_script_package_accepts_extra_files():
    package = bridge.build_script_package(files={"data/corpus.txt": "hello grid", "tool.py": b"X = 1\n"})
    with tarfile.open(fileobj=io.BytesIO(package), mode="r:gz") as tar:
        assert tar.extractfile("data/corpus.txt").read() == b"hello grid"


def _gz_tar(entries):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, payload in entries.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


@pytest.mark.parametrize(
    "package,fragment",
    [
        (b"", "empty script package"),
        (b"x" * (bridge.MAX_SCRIPT_PACKAGE_BYTES + 1), "too large"),
        (b"not a tarball", "invalid tarball"),
        (_gz_tar({"train.py": b"print('hi')\n"}), "run.py"),
        (_gz_tar({"run.py": b"pass\n", ".env": b"SECRET=1"}), "forbidden path"),
        (_gz_tar({"run.py": b"pass\n", "keys/id_rsa": b"k"}), "forbidden path"),
        (_gz_tar({"run.py": b"pass\n", "cert.pem": b"c"}), "forbidden path"),
        (_gz_tar({"run.py": b"pass\n", "../escape.py": b"x"}), "Invalid tar member"),
    ],
)
def test_validate_script_package_rejections(package, fragment):
    with pytest.raises(ValueError, match=fragment):
        bridge.validate_script_package(package)


def test_validate_script_package_rejects_symlinks():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        run = tarfile.TarInfo(name="run.py")
        run.size = 5
        tar.addfile(run, io.BytesIO(b"pass\n"))
        link = tarfile.TarInfo(name="link.py")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tar.addfile(link)
    with pytest.raises(ValueError, match="Invalid tar member"):
        bridge.validate_script_package(buffer.getvalue())


# ---------------------------------------------------------------------------
# End-to-end: the exact ScriptRunner task flow on a worker node
# ---------------------------------------------------------------------------


def test_run_py_trains_and_writes_worker_outputs(tmp_path):
    arch = dict(bridge.GRID_PROFILES["mmn-tiny"])
    arch.update({"dim": 16, "n_unique_layers": 1, "n_logical_layers": 1, "ffn_dim": 32, "seq_len": 64})
    package = bridge.build_script_package(
        config={"job_id": "job-test-1", "job_type": "train", "architecture_config": arch},
        dataset=[{"input": "hi", "output": "yo"}, {"input": "ping", "output": "pong"}],
    )

    task_dir = tmp_path / "task-001"
    task_dir.mkdir()
    with tarfile.open(fileobj=io.BytesIO(package), mode="r:gz") as tar:
        tar.extractall(task_dir, filter="data")
    hyperparams = {"epochs": 2, "batch_size": 2, "lr": 3e-4, "vocab_size": 96, "seed": 5}
    (task_dir / "hyperparams.json").write_text(json.dumps(hyperparams), encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "DISTRIBAI_TASK_ID": "task-001",
            "DISTRIBAI_JOB_ID": "job-test-1",
            "DISTRIBAI_JOB_TYPE": "train",
            "PYTHONPATH": str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", ""),
        }
    )
    result = subprocess.run(
        [project_python(), "run.py"],
        cwd=task_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr

    results = json.loads((task_dir / "results.json").read_text(encoding="utf-8"))
    assert results["task_id"] == "task-001"
    assert results["job_id"] == "job-test-1"
    assert results["framework"] == "magicmindnet"
    assert len(results["losses"]) == 2
    metrics = json.loads((task_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["loss"] == results["final_loss"]

    # The checkpoint a worker uploads must read back into a Chatbot.
    imported = bridge.import_checkpoint(str(task_dir / "checkpoint.pt"))
    assert imported.vocab_size == 96
    assert imported.d_model == 16
