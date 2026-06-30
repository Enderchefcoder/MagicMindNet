from pathlib import Path

import magicmindnet as ai
from conftest import load_checkpoint, tensor_entry_first_f32


def test_merge_classifier_averages_backbone_weight(tmp_path: Path):
    labels_path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "labels_small.json"
    ds = ai.DatasetClassification(str(labels_path), "text", "tag")
    a = ai.Classifier.from_classification(ds, input_dim=16, seed=1)
    b = ai.Classifier.from_classification(ds, input_dim=16, seed=2)
    path_a = tmp_path / "a.mmn"
    path_b = tmp_path / "b.mmn"
    path_m = tmp_path / "m.mmn"
    ai.export_classifier(a, "safetensors", str(path_a))
    ai.export_classifier(b, "safetensors", str(path_b))
    merged = ai.merge_classifier(a, b)
    ai.export_classifier(merged, "safetensors", str(path_m))
    wa = tensor_entry_first_f32(load_checkpoint(path_a)["tensors"]["backbone"])
    wb = tensor_entry_first_f32(load_checkpoint(path_b)["tensors"]["backbone"])
    wm = tensor_entry_first_f32(load_checkpoint(path_m)["tensors"]["backbone"])
    assert abs(wm - (wa + wb) / 2.0) < 1e-5


def test_merge_classifier_averages_head_weight(tmp_path: Path):
    labels_path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "labels_small.json"
    ds = ai.DatasetClassification(str(labels_path), "text", "tag")
    a = ai.Classifier.from_classification(ds, input_dim=16, seed=1)
    b = ai.Classifier.from_classification(ds, input_dim=16, seed=2)
    path_a = tmp_path / "a.mmn"
    path_b = tmp_path / "b.mmn"
    path_m = tmp_path / "m.mmn"
    ai.export_classifier(a, "safetensors", str(path_a))
    ai.export_classifier(b, "safetensors", str(path_b))
    merged = ai.merge_classifier(a, b)
    ai.export_classifier(merged, "safetensors", str(path_m))
    wa = tensor_entry_first_f32(load_checkpoint(path_a)["tensors"]["head"])
    wb = tensor_entry_first_f32(load_checkpoint(path_b)["tensors"]["head"])
    wm = tensor_entry_first_f32(load_checkpoint(path_m)["tensors"]["head"])
    assert abs(wm - (wa + wb) / 2.0) < 1e-5
