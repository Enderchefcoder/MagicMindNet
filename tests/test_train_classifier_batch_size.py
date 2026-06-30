"""TrainClassifier honors TrainConfig.batch_size via gradient accumulation."""

import json
from pathlib import Path

import magicmindnet as ai


def test_train_classifier_batch_size_two_reduces_mean_loss(tmp_path: Path):
    cls_path = tmp_path / "cls.json"
    cls_path.write_text(
        json.dumps(
            [
                {"text": "sun", "tag": "Happy"},
                {"text": "rain", "tag": "Sad"},
                {"text": "bright", "tag": "Happy"},
            ]
        ),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(str(cls_path), "text", "tag")
    clf = ai.Classifier.from_classification(ds, input_dim=32, seed=42)
    before = clf.compute_mean_loss(ds)
    cfg = ai.TrainConfig(epochs=20, batch_size=2, learning_rate=0.05, optimizer="adamw")
    ai.TrainClassifier(clf, ds, cfg)
    after = clf.compute_mean_loss(ds)
    assert after < before, f"mean loss before={before} after={after}"
