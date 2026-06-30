import json
from pathlib import Path

import pytest

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_dataset_corpus_two_files(tmp_path):
    rowfile = tmp_path / "corp.json"
    rowfile.write_text(
        json.dumps([{"text": "short"}, {"text": "a much longer piece of text"}]),
        encoding="utf-8",
    )
    txtfile = tmp_path / "corpus.txt"
    txtfile.write_text("word " * 100, encoding="utf-8")
    ds = ai.DatasetCorpus(
        use_two_files=True,
        rowfile=str(rowfile),
        txtfile=str(txtfile),
        sort_rows_by_complexity=True,
    )
    assert ds.rows >= 2


def test_dataset_classification(tmp_path):
    path = tmp_path / "cls.json"
    path.write_text(
        json.dumps([{"text": "Yay!", "tags": "Happy"}]),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(file=str(path), text_col="text", tags_col="tags")
    assert ds.rows == 1


def test_diffusion_and_classifier_construct():
    d = ai.Diffusion()
    assert d is not None
    c = ai.Classifier(num_labels=3, input_dim=16)
    pred = c.predict("hello")
    assert len(pred) == 3
    assert sum(pred.values()) == pytest.approx(1.0, rel=1e-3)
