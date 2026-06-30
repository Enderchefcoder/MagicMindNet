"""Classifier label wiring from classification datasets."""

import json

import magicmindnet as ai


def test_classifier_from_classification_uses_tag_labels(tmp_path):
    path = tmp_path / "cls.json"
    path.write_text(
        json.dumps(
            [
                {"text": "I am happy", "tag": "Happy"},
                {"text": "I am sad", "tag": "Sad"},
            ]
        ),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(str(path), "text", "tag")
    clf = ai.Classifier.from_classification(ds, input_dim=32)
    assert sorted(clf.labels) == ["Happy", "Sad"]
    pred = clf.predict("cheerful day")
    assert "Happy" in pred
    assert "Sad" in pred


def test_classifier_with_labels_explicit():
    clf = ai.Classifier.with_labels(["cat", "dog"], input_dim=16)
    assert clf.labels == ["cat", "dog"]
