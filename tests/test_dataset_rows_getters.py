from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_dataset_qa_rows_and_type_getters():
    ds = ai.DatasetQA(file=str(FIXTURES / "qa_valid.json"), user_row="input", ai_row="output")
    assert ds.rows == 2
    assert ds.type_ == "qa"
    assert ds.format


def test_dataset_classification_rows_and_unique_labels():
    ds = ai.DatasetClassification(
        str(FIXTURES / "labels_small.json"), "text", "tag"
    )
    assert ds.rows == 2
    assert ds.type_ == "classification"
    assert sorted(ds.unique_labels()) == ["neg", "pos"]
