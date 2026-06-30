from pathlib import Path

import magicmindnet as ai


def test_classifier_nested_export_import_roundtrip(tmp_path: Path):
    clf = ai.Classifier.with_labels(["pos", "neg"], input_dim=24, seed=5)
    path = tmp_path / "nested" / "clf.mmn"
    ai.export_classifier(clf, "safetensors", str(path))
    loaded = ai.import_classifier("safetensors", [str(path)])
    assert loaded.labels == ["pos", "neg"]
    assert loaded.input_dim == 24
    assert loaded.init_seed == 5
