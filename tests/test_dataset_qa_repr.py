from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_dataset_qa_repr_shows_rows_and_format():
    ds = ai.DatasetQA(file=str(FIXTURES / "qa_valid.json"), user_row="input", ai_row="output")
    text = repr(ds)
    assert "DatasetQA" in text
    assert "rows=" in text
    assert str(ds.rows) in text
