import json
from pathlib import Path

import pytest

import magicmindnet as ai


def _export_classifier(tmp_path: Path) -> Path:
    clf = ai.Classifier.with_labels(["a", "b"], input_dim=16, seed=1)
    path = tmp_path / "clf.mmn"
    ai.export_classifier(clf, "safetensors", str(path))
    return path


def test_import_classifier_rejects_missing_input_dim_py(tmp_path: Path):
    path = _export_classifier(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["meta"].pop("input_dim", None)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_classifier("safetensors", [str(path)])
    assert "input_dim" in str(exc.value).lower()


def test_import_classifier_rejects_missing_backbone_py(tmp_path: Path):
    path = _export_classifier(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tensors"].pop("backbone", None)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_classifier("safetensors", [str(path)])
    assert "backbone" in str(exc.value).lower()


def test_import_classifier_rejects_empty_labels_py(tmp_path: Path):
    path = _export_classifier(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["meta"]["labels"] = []
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_classifier("safetensors", [str(path)])
    assert "label" in str(exc.value).lower()


def test_import_classifier_rejects_invalid_json_py(tmp_path: Path):
    path = tmp_path / "bad.mmn"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)):
        ai.import_classifier("safetensors", [str(path)])


def test_import_classifier_rejects_empty_file_py(tmp_path: Path):
    path = tmp_path / "empty.mmn"
    path.write_text("", encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)):
        ai.import_classifier("safetensors", [str(path)])


def test_import_classifier_rejects_backbone_shape_mismatch_py(tmp_path: Path):
    path = _export_classifier(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["meta"]["input_dim"] = 32
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_classifier("safetensors", [str(path)])
    msg = str(exc.value).lower()
    assert "backbone" in msg and "shape" in msg


def test_import_classifier_rejects_head_shape_mismatch_py(tmp_path: Path):
    path = _export_classifier(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["meta"]["labels"] = ["a", "b", "c"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_classifier("safetensors", [str(path)])
    msg = str(exc.value).lower()
    assert "head" in msg and "shape" in msg
