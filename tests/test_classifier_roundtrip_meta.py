import json
from pathlib import Path

import magicmindnet as ai


def test_classifier_roundtrip_preserves_input_dim_and_num_labels(tmp_path: Path):
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            [
                {"text": "a", "tag": "X"},
                {"text": "b", "tag": "Y"},
                {"text": "c", "tag": "Z"},
            ]
        ),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(str(path), "text", "tag")
    clf = ai.Classifier.from_classification(ds, input_dim=40, seed=3)
    ckpt = tmp_path / "clf.mmn"
    ai.export_classifier(clf, "safetensors", str(ckpt))
    loaded = ai.import_classifier("safetensors", [str(ckpt)])
    assert loaded.input_dim == 40
    assert loaded.num_labels == 3
    assert loaded.labels == clf.labels
