from pathlib import Path

import pytest

import magicmindnet as ai


def test_import_classifier_rejects_unknown_format(tmp_path: Path):
    clf = ai.Classifier.with_labels(["a", "b"], input_dim=8, seed=1)
    path = tmp_path / "clf.mmn"
    ai.export_classifier(clf, "safetensors", str(path))
    with pytest.raises((ValueError, RuntimeError)):
        ai.import_classifier("onnx", [str(path)])
