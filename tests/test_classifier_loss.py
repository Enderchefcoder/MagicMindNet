import json

import magicmindnet as ai


def test_classifier_compute_loss_decreases_after_train(tmp_path):
    path = tmp_path / "cls.json"
    path.write_text(
        json.dumps(
            [
                {"text": "good day", "tag": "Happy"},
                {"text": "great day", "tag": "Happy"},
            ]
        ),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(str(path), "text", "tag")
    clf = ai.Classifier.from_classification(ds, input_dim=32)
    before = clf.compute_loss("good day", "Happy")
    cfg = ai.TrainConfig(epochs=8, learning_rate=0.08)
    ai.TrainClassifier(clf, ds, cfg)
    after = clf.compute_loss("good day", "Happy")
    assert after <= before, f"loss before={before} after={after}"
