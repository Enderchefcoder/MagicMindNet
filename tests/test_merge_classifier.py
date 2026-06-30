import json

import pytest

import magicmindnet as ai


def test_merge_classifier_averages_identical_models(tmp_path):
    path = tmp_path / "cls.json"
    path.write_text(
        json.dumps(
            [
                {"text": "a", "tag": "X"},
                {"text": "b", "tag": "Y"},
            ]
        ),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(str(path), "text", "tag")
    a = ai.Classifier.from_classification(ds, input_dim=32, seed=42)
    b = ai.Classifier.from_classification(ds, input_dim=32, seed=42)
    m = ai.merge_classifier(a, b)
    assert m.labels == a.labels
    pred_a = a.predict("a")
    pred_m = m.predict("a")
    for label in a.labels:
        assert abs(pred_a[label] - pred_m[label]) < 1e-4


def test_merge_classifier_rejects_label_mismatch():
    a = ai.Classifier.with_labels(["cat"], input_dim=16)
    b = ai.Classifier.with_labels(["dog"], input_dim=16)
    with pytest.raises(ai.ModelMismatchError):
        ai.merge_classifier(a, b)
