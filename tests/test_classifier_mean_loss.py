import json

import pytest

import magicmindnet as ai


def test_classifier_compute_mean_loss_decreases_after_train(tmp_path):
    path = tmp_path / "cls.json"
    path.write_text(
        json.dumps(
            [
                {"text": "sun", "tag": "Happy"},
                {"text": "rain", "tag": "Sad"},
                {"text": "bright", "tag": "Happy"},
            ]
        ),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(str(path), "text", "tag")
    clf = ai.Classifier.from_classification(ds, input_dim=32, seed=42)
    before = clf.compute_mean_loss(ds)
    assert before > 0.0
    cfg = ai.TrainConfig(epochs=20, learning_rate=0.01, optimizer="adamw")
    ai.TrainClassifier(clf, ds, cfg)
    after = clf.compute_mean_loss(ds)
    assert after < before, f"mean classification loss before={before} after={after}"


def test_classifier_mean_loss_wrong_dataset_type(tmp_path):
    qa = tmp_path / "qa.json"
    qa.write_text(
        json.dumps([{"input": "hi", "output": "yo"}]),
        encoding="utf-8",
    )
    ds_qa = ai.DatasetQA(str(qa), "input", "output")
    clf = ai.Classifier.with_labels(["A"], input_dim=16, seed=1)
    with pytest.raises(ai.DataMismatchError):
        clf.compute_mean_loss(ds_qa)
