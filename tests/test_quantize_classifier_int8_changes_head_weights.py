import json
from pathlib import Path

import magicmindnet as ai


def test_quantize_classifier_int8_changes_head_weights(tmp_path: Path):
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "labels_small.json"
    ds = ai.DatasetClassification(str(path), "text", "tag")
    clf = ai.Classifier.from_classification(ds, input_dim=16, seed=1)
    before_path = tmp_path / "before.mmn"
    ai.export_classifier(clf, "safetensors", str(before_path))
    before = json.loads(before_path.read_text(encoding="utf-8"))["tensors"]["head"]["data"]
    ai.quantize_classifier(clf, "int8")
    after_path = tmp_path / "after.mmn"
    ai.export_classifier(clf, "safetensors", str(after_path))
    after = json.loads(after_path.read_text(encoding="utf-8"))["tensors"]["head"]["data"]
    assert before != after
