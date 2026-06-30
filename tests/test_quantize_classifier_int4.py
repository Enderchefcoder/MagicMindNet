from pathlib import Path

import magicmindnet as ai


def test_quantize_classifier_int4_runs():
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "labels_small.json"
    ds = ai.DatasetClassification(str(path), "text", "tag")
    clf = ai.Classifier.from_classification(ds, input_dim=16, seed=1)
    before = clf.predict("happy")
    ai.quantize_classifier(clf, "int4")
    after = clf.predict("happy")
    assert set(before.keys()) == set(after.keys())
