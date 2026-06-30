"""Parametric coverage of the classifier safetensors IO contract (see docs/checkpoint_coverage.md)."""

import json
from pathlib import Path

import pytest

import magicmindnet as ai
from conftest import load_checkpoint_tensors, tensor_entry_first_f32

CLASSIFIER_TENSOR_KEYS = ["backbone", "head"]

# Same element count as exported tensors (input_dim=16, 2 labels, hidden=128).
WRONG_SHAPES = {
    "backbone": [64, 32],
    "head": [4, 64],
}


def _export_classifier(tmp_path: Path) -> Path:
    clf = ai.Classifier.with_labels(["a", "b"], input_dim=16, seed=1)
    path = tmp_path / "clf.mmn"
    ai.export_classifier(clf, "safetensors", str(path))
    return path


@pytest.mark.parametrize("tensor_key", CLASSIFIER_TENSOR_KEYS)
def test_import_classifier_rejects_missing_tensor_matrix_py(
    tmp_path: Path, tensor_key: str
):
    path = _export_classifier(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tensors"].pop(tensor_key, None)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_classifier("safetensors", [str(path)])
    assert tensor_key.lower() in str(exc.value).lower()


@pytest.mark.parametrize("tensor_key", CLASSIFIER_TENSOR_KEYS)
def test_import_classifier_rejects_shape_mismatch_matrix_py(
    tmp_path: Path, tensor_key: str
):
    path = _export_classifier(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tensors"][tensor_key]["shape"] = WRONG_SHAPES[tensor_key]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_classifier("safetensors", [str(path)])
    msg = str(exc.value).lower()
    assert tensor_key.lower() in msg and "shape" in msg


@pytest.mark.parametrize("tensor_key", CLASSIFIER_TENSOR_KEYS)
def test_merge_classifier_averages_tensor_matrix_py(tmp_path: Path, tensor_key: str):
    a = ai.Classifier.with_labels(["a", "b"], input_dim=16, seed=1)
    b = ai.Classifier.with_labels(["a", "b"], input_dim=16, seed=2)
    path_a = tmp_path / "a.mmn"
    path_b = tmp_path / "b.mmn"
    path_m = tmp_path / "m.mmn"
    ai.export_classifier(a, "safetensors", str(path_a))
    ai.export_classifier(b, "safetensors", str(path_b))
    merged = ai.merge_classifier(a, b)
    ai.export_classifier(merged, "safetensors", str(path_m))
    tensors_a = load_checkpoint_tensors(path_a)
    tensors_b = load_checkpoint_tensors(path_b)
    tensors_m = load_checkpoint_tensors(path_m)
    wa = tensor_entry_first_f32(tensors_a[tensor_key])
    wb = tensor_entry_first_f32(tensors_b[tensor_key])
    wm = tensor_entry_first_f32(tensors_m[tensor_key])
    assert abs(wm - (wa + wb) / 2.0) < 1e-5


@pytest.mark.parametrize("mode", ["int8", "int4"])
@pytest.mark.parametrize("tensor_key", CLASSIFIER_TENSOR_KEYS)
def test_quantize_classifier_changes_tensor_matrix_py(
    tmp_path: Path, mode: str, tensor_key: str
):
    clf = ai.Classifier.with_labels(["a", "b"], input_dim=16, seed=1)
    before_path = tmp_path / f"before_{mode}_{tensor_key}.mmn"
    ai.export_classifier(clf, "safetensors", str(before_path))
    before = load_checkpoint_tensors(before_path)
    ai.quantize_classifier(clf, mode)
    after_path = tmp_path / f"after_{mode}_{tensor_key}.mmn"
    ai.export_classifier(clf, "safetensors", str(after_path))
    after = load_checkpoint_tensors(after_path)
    assert before[tensor_key]["data"] != after[tensor_key]["data"]
