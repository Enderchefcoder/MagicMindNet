import json

import magicmindnet as ai


def test_classifier_seed_is_deterministic(tmp_path):
    path = tmp_path / "cls.json"
    path.write_text(
        json.dumps([{"text": "hi", "tag": "A"}, {"text": "bye", "tag": "B"}]),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(str(path), "text", "tag")
    a = ai.Classifier.from_classification(ds, input_dim=32, seed=7)
    b = ai.Classifier.from_classification(ds, input_dim=32, seed=7)
    assert a.compute_loss("hi", "A") == b.compute_loss("hi", "A")


def test_train_config_getters():
    cfg = ai.TrainConfig(
        epochs=3,
        batch_size=4,
        cuda=False,
        optimizer="adamw",
        learning_rate=0.01,
    )
    assert cfg.epochs == 3
    assert cfg.batch_size == 4
    assert cfg.cuda is False
    assert cfg.optimizer == "adamw"
    assert abs(cfg.learning_rate - 0.01) < 1e-9
