from pathlib import Path

import pytest

import magicmindnet as ai


def _tiny_classifier():
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "labels_small.json"
    ds = ai.DatasetClassification(str(path), "text", "tag")
    return ai.Classifier.from_classification(ds, input_dim=16, seed=1)


def test_quantize_classifier_int8_runs():
    clf = _tiny_classifier()
    before = clf.predict("happy")
    ai.quantize_classifier(clf, "int8")
    after = clf.predict("happy")
    assert set(before.keys()) == set(after.keys())


def test_quantize_classifier_rejects_unknown_mode():
    clf = _tiny_classifier()
    with pytest.raises(RuntimeError, match="Unknown quant"):
        ai.quantize_classifier(clf, "fp16")
