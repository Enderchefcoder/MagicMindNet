import tomllib
from pathlib import Path

import magicmindnet as ai


def test_version_matches_pyproject():
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert ai.__version__ == data["project"]["version"]
