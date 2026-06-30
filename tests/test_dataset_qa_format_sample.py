import json
from pathlib import Path

import pytest

import magicmindnet as ai


def test_format_sample_returns_chatxml_string(tmp_path: Path):
    path = tmp_path / "qa.json"
    path.write_text(
        json.dumps([{"input": "Hi", "output": "Hello"}]),
        encoding="utf-8",
    )
    ds = ai.DatasetQA(file=str(path), user_row="input", ai_row="output")
    text = ds.format_sample(0)
    assert isinstance(text, str)
    assert len(text) > 0


def test_format_sample_out_of_range_raises(tmp_path: Path):
    path = tmp_path / "qa.json"
    path.write_text(json.dumps([{"input": "a", "output": "b"}]), encoding="utf-8")
    ds = ai.DatasetQA(file=str(path), user_row="input", ai_row="output")
    with pytest.raises(ValueError, match="out of range"):
        ds.format_sample(99)
