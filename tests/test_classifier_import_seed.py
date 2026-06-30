from pathlib import Path

import magicmindnet as ai


def test_classifier_import_restores_init_seed_getter(tmp_path: Path):
    clf = ai.Classifier.with_labels(["x", "y"], input_dim=16, seed=88)
    path = tmp_path / "clf.mmn"
    ai.export_classifier(clf, "safetensors", str(path))
    loaded = ai.import_classifier("safetensors", [str(path)])
    assert loaded.init_seed == 88
