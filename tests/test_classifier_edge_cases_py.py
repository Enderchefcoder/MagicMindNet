"""Classifier edge cases: single label, unknown tags in mean loss, empty text."""

import json
import math

import magicmindnet as ai


def test_single_label_classifier_predict_sums_to_one():
    clf = ai.Classifier.with_labels(["only"], input_dim=16, seed=3)
    pred = clf.predict("anything")
    assert list(pred.keys()) == ["only"]
    assert abs(sum(pred.values()) - 1.0) < 1e-4


def test_compute_mean_loss_skips_unknown_tags(tmp_path):
    path = tmp_path / "mixed.json"
    path.write_text(
        json.dumps(
            [
                {"text": "good", "tag": "pos"},
                {"text": "bad", "tag": "neg"},
                {"text": "weird", "tag": "orphan"},
            ]
        ),
        encoding="utf-8",
    )
    known_path = tmp_path / "known.json"
    known_path.write_text(
        json.dumps(
            [
                {"text": "good", "tag": "pos"},
                {"text": "bad", "tag": "neg"},
            ]
        ),
        encoding="utf-8",
    )
    ds_mixed = ai.DatasetClassification(str(path), "text", "tag")
    ds_known = ai.DatasetClassification(str(known_path), "text", "tag")
    clf = ai.Classifier.with_labels(["pos", "neg"], input_dim=16, seed=1)
    loss_mixed = clf.compute_mean_loss(ds_mixed)
    loss_known = clf.compute_mean_loss(ds_known)
    assert math.isfinite(loss_mixed)
    assert abs(loss_mixed - loss_known) < 1e-5


def test_predict_empty_string_returns_all_labels():
    clf = ai.Classifier.with_labels(["a", "b"], input_dim=8, seed=2)
    pred = clf.predict("")
    assert set(pred.keys()) == {"a", "b"}
    assert abs(sum(pred.values()) - 1.0) < 1e-4


def test_hybrid_train_classifier_reduces_loss(tmp_path):
    path = tmp_path / "cls.json"
    path.write_text(
        json.dumps(
            [
                {"text": "great", "tag": "pos"},
                {"text": "terrible", "tag": "neg"},
                {"text": "nice", "tag": "pos"},
            ]
        ),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(str(path), "text", "tag")
    clf = ai.Classifier.from_classification(ds, input_dim=32, seed=7)
    cfg = ai.TrainConfig(epochs=5, batch_size=2, learning_rate=0.08, optimizer="hybrid")
    before = clf.compute_mean_loss(ds)
    ai.TrainClassifier(clf, ds, cfg)
    after = clf.compute_mean_loss(ds)
    assert after < before
