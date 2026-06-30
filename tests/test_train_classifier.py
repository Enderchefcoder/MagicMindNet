"""Classifier training API."""

import json
from pathlib import Path

import magicmindnet as ai


def test_train_classifier_completes_and_predicts(tmp_path: Path):
    path = tmp_path / "cls.json"
    path.write_text(
        json.dumps(
            [
                {"text": "great day", "tag": "Happy"},
                {"text": "awful day", "tag": "Sad"},
            ]
        ),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(str(path), "text", "tag")
    clf = ai.Classifier.from_classification(ds, input_dim=32)
    cfg = ai.TrainConfig(epochs=3, batch_size=1, learning_rate=0.05)
    ai.TrainClassifier(clf, ds, cfg)
    pred = clf.predict("great day")
    assert "Happy" in pred
    assert "Sad" in pred
    assert abs(sum(pred.values()) - 1.0) < 1e-3
