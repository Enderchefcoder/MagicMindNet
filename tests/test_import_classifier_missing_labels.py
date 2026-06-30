import json
from pathlib import Path

import pytest

import magicmindnet as ai


def test_import_classifier_rejects_missing_labels_meta(tmp_path: Path):
    clf = ai.Classifier.with_labels(["pos", "neg"], input_dim=12, seed=2)
    path = tmp_path / "clf.mmn"
    ai.export_classifier(clf, "safetensors", str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["meta"].pop("labels", None)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_classifier("safetensors", [str(path)])
    assert "label" in str(exc.value).lower()
