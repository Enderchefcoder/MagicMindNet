"""Train a small classifier and save a checkpoint."""

import json
import tempfile
from pathlib import Path

import magicmindnet as ai

data = [
    {"text": "I feel great today", "tag": "Happy"},
    {"text": "This is wonderful", "tag": "Happy"},
    {"text": "I am upset", "tag": "Sad"},
    {"text": "Everything is awful", "tag": "Sad"},
]

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "labels.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    ds = ai.DatasetClassification(str(path), "text", "tag")
    clf = ai.Classifier.from_classification(ds, input_dim=64)
    cfg = ai.TrainConfig(epochs=8, learning_rate=0.08)
    ai.TrainClassifier(clf, ds, cfg)
    ckpt = Path(tmp) / "classifier.mmn"
    ai.export_classifier(clf, "safetensors", str(ckpt))
    loaded = ai.import_classifier("safetensors", [str(ckpt)])
    print("labels:", loaded.labels)
    print("predict:", loaded.predict("I feel great today"))
