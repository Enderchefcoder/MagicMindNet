"""In-memory datasets: `DatasetQA(data=[...])` / `DatasetClassification(data=[...])`."""

import pytest

import magicmindnet as ai

QA_ROWS = [
    {"input": "hi", "output": "hello"},
    {"input": "bye", "output": "goodbye"},
]

CLS_ROWS = [
    {"text": "sunny day", "label": "happy"},
    {"text": "rainy day", "label": "sad"},
]


def test_dataset_qa_from_memory_rows_and_format():
    ds = ai.DatasetQA(data=QA_ROWS)
    assert ds.rows == 2
    assert ds.format == "memory"
    assert ds.type_ == "qa"


def test_dataset_qa_from_memory_format_sample_uses_chatxml():
    ds = ai.DatasetQA(data=QA_ROWS)
    text = ds.format_sample(0)
    assert "hi" in text and "hello" in text


def test_dataset_qa_from_memory_custom_rows():
    ds = ai.DatasetQA(data=[{"q": "one", "a": "two"}], user_row="q", ai_row="a")
    assert ds.rows == 1
    assert "one" in ds.format_sample(0)


def test_dataset_qa_from_memory_system_row():
    ds = ai.DatasetQA(
        data=[{"input": "x", "output": "y", "sys": "be brief"}],
        system_row="sys",
    )
    assert "be brief" in ds.format_sample(0)


def test_dataset_qa_memory_missing_key_raises_data_missing_row():
    with pytest.raises(ai.DataMissingRowError, match="output"):
        ai.DatasetQA(data=[{"input": "only input"}])


def test_dataset_qa_requires_file_or_data():
    with pytest.raises(ValueError, match="file"):
        ai.DatasetQA()


def test_dataset_qa_rejects_file_and_data_together(tmp_path):
    f = tmp_path / "qa.json"
    f.write_text('[{"input": "a", "output": "b"}]', encoding="utf-8")
    with pytest.raises(ValueError, match="not both"):
        ai.DatasetQA(file=str(f), data=QA_ROWS)


def test_dataset_classification_from_memory():
    ds = ai.DatasetClassification(data=CLS_ROWS)
    assert ds.rows == 2
    assert ds.format == "memory"
    assert ds.unique_labels() == ["happy", "sad"]


def test_dataset_classification_from_memory_custom_columns():
    ds = ai.DatasetClassification(
        data=[{"body": "great", "tag": "pos"}], text_col="body", tags_col="tag"
    )
    assert ds.unique_labels() == ["pos"]


def test_dataset_classification_memory_missing_key_raises():
    with pytest.raises(ai.DataMissingRowError, match="label"):
        ai.DatasetClassification(data=[{"text": "no label here"}])


def test_dataset_classification_requires_file_or_data():
    with pytest.raises(ValueError, match="file"):
        ai.DatasetClassification()


def test_memory_qa_dataset_trains_chatbot():
    ds = ai.DatasetQA(data=QA_ROWS)
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=3)
    losses = bot.train(ds, epochs=2, learning_rate=0.05, optimizer="adamw")
    assert len(losses) == 2


def test_memory_classification_dataset_trains_classifier():
    ds = ai.DatasetClassification(data=CLS_ROWS)
    clf = ai.Classifier.from_classification(ds, input_dim=16, seed=3)
    losses = clf.train(ds, epochs=2, learning_rate=0.05)
    assert len(losses) == 2
