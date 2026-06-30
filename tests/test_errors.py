from pathlib import Path

import pytest

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_dataset_qa_missing_input_row_raises():
    path = FIXTURES / "qa_no_input.json"
    with pytest.raises(ai.DataMissingRowError) as exc:
        ai.DatasetQA(file=str(path), user_row="input", ai_row="output")
    assert "input" in str(exc.value)


def test_dataset_qa_loads_valid():
    path = FIXTURES / "qa_valid.json"
    ds = ai.DatasetQA(file=str(path), user_row="input", ai_row="output")
    assert ds.rows == 2
    assert ds.format == "json"
    assert ds.type_ == "qa"


def test_cpu_error_has_message():
    err = ai.CPUError("CPU not accessible: test")
    assert "CPU" in str(err)
