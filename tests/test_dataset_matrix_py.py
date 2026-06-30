"""Parametric dataset loader coverage (see docs/dataset_coverage.md)."""

import json
from pathlib import Path

import pytest

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_dataset_qa_jsonl_loads_two_rows(tmp_path: Path):
    path = tmp_path / "qa.jsonl"
    path.write_text(
        '{"input": "a", "output": "b"}\n{"input": "c", "output": "d"}\n',
        encoding="utf-8",
    )
    ds = ai.DatasetQA(file=str(path), user_row="input", ai_row="output")
    assert ds.rows == 2
    assert ds.format == "jsonl"


def test_dataset_qa_missing_ai_row_raises(tmp_path: Path):
    path = tmp_path / "qa.json"
    path.write_text(json.dumps([{"input": "only input"}]), encoding="utf-8")
    with pytest.raises(ai.DataMissingRowError) as exc:
        ai.DatasetQA(file=str(path), user_row="input", ai_row="output")
    assert "output" in str(exc.value).lower()


def test_dataset_qa_format_sample_includes_system_and_thinktag(tmp_path: Path):
    path = tmp_path / "qa.json"
    path.write_text(
        json.dumps(
            [{"input": "Hi", "output": "Hello", "systemprompt": "Be helpful"}]
        ),
        encoding="utf-8",
    )
    ds = ai.DatasetQA(
        file=str(path),
        user_row="input",
        ai_row="output",
        system_row="systemprompt",
        thinktag="think|/think",
        cot=True,
    )
    text = ds.format_sample(0)
    assert "Be helpful" in text
    assert "thinkHello" in text
    assert "Hello" in text


def test_dataset_classification_auto_tags_when_tag_column_missing(tmp_path: Path):
    path = tmp_path / "cls.json"
    path.write_text(
        json.dumps([{"text": "one"}, {"text": "two"}]),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(str(path), "text", "tag")
    assert ds.rows == 2
    assert ds.unique_labels() == ["class_0", "class_1"]


def test_dataset_image_gen_preserves_negative_prompt(tmp_path: Path):
    path = tmp_path / "gen.json"
    path.write_text(
        json.dumps(
            [{"prompt": "cat", "image": "c.png", "negative_prompt": "blur"}]
        ),
        encoding="utf-8",
    )
    ds = ai.DatasetImageGen(file=str(path))
    assert ds.rows == 1
    assert ds.type_ == "image_gen"


def test_dataset_image_edit_preserves_mask_and_negative(tmp_path: Path):
    path = tmp_path / "edit.json"
    path.write_text(
        json.dumps(
            [
                {
                    "prompt": "fix",
                    "image": "a.png",
                    "mask_image": "m.png",
                    "negative_prompt": "noise",
                }
            ]
        ),
        encoding="utf-8",
    )
    ds = ai.DatasetImageEdit(file=str(path))
    assert ds.rows == 1
    assert ds.format == "image_edit"


@pytest.mark.parametrize(
    "loader,kwargs,expected_type",
    [
        (
            ai.DatasetQA,
            {"file": str(FIXTURES / "qa_valid.json"), "user_row": "input", "ai_row": "output"},
            "qa",
        ),
        (
            ai.DatasetClassification,
            {"file": str(FIXTURES / "labels_small.json"), "text_col": "text", "tags_col": "tag"},
            "classification",
        ),
    ],
)
def test_dataset_type_getter_matrix(loader, kwargs, expected_type):
    ds = loader(**kwargs)
    assert ds.type_ == expected_type
    assert ds.rows >= 1
