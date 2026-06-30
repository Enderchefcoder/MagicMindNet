from pathlib import Path

import magicmindnet as ai


def test_export_classifier_creates_nested_output_file(tmp_path: Path):
    clf = ai.Classifier.with_labels(["A", "B"], input_dim=16, seed=1)
    out = tmp_path / "nested" / "clf.mmn"
    ai.export_classifier(clf, "safetensors", str(out))
    assert out.is_file()
    assert out.stat().st_size > 0
