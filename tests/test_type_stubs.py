"""Typing support ships with the package: `py.typed` marker + `_native.pyi` stubs."""

from pathlib import Path

import magicmindnet


def package_dir() -> Path:
    return Path(magicmindnet.__file__).parent


def test_py_typed_marker_ships_with_package():
    assert (package_dir() / "py.typed").is_file()


def test_native_stub_file_ships_with_package():
    assert (package_dir() / "_native.pyi").is_file()


def test_stub_covers_public_native_names():
    stub = (package_dir() / "_native.pyi").read_text(encoding="utf-8")
    for name in (
        "class Chatbot",
        "class Classifier",
        "class Diffusion",
        "class TrainConfig",
        "class DatasetQA",
        "def Train",
        "def load",
        "def export",
    ):
        assert name in stub, f"stub missing: {name}"
