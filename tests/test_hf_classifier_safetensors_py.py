"""Hugging Face binary safetensors for Classifier."""

from pathlib import Path

import pytest

import magicmindnet as ai


def test_hf_classifier_safetensors_roundtrip(tmp_path: Path):
    clf = ai.Classifier.with_labels(["Happy", "Sad"], input_dim=32)
    path = tmp_path / "clf.safetensors"
    ai.export_classifier(clf, "hf-safetensors", str(path))
    raw = path.read_bytes()
    assert not raw.startswith(b"{")
    loaded = ai.import_classifier("hf-safetensors", [str(path)])
    assert loaded.labels == clf.labels
    assert loaded.input_dim == clf.input_dim
    before = clf.predict("hello")
    after = loaded.predict("hello")
    assert before == pytest.approx(after)


def test_import_classifier_auto_detects_hf_binary(tmp_path: Path):
    clf = ai.Classifier.with_labels(["a", "b"], input_dim=16)
    path = tmp_path / "clf_auto.safetensors"
    ai.export_classifier(clf, "hf-safetensors", str(path))
    loaded = ai.import_classifier("safetensors", [str(path)])
    assert loaded.labels == clf.labels
