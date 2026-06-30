"""Classifier training benchmark with before/after loss (seeded)."""

import json
import tempfile
from pathlib import Path

import magicmindnet as ai

DATA = [
    {"text": "great day", "tag": "Happy"},
    {"text": "awful day", "tag": "Sad"},
    {"text": "nice day", "tag": "Happy"},
]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "labels.json"
        path.write_text(json.dumps(DATA), encoding="utf-8")
        ds = ai.DatasetClassification(str(path), "text", "tag")
        clf = ai.Classifier.from_classification(ds, input_dim=32, seed=42)
        before = clf.compute_loss("great day", "Happy")
        cfg = ai.TrainConfig(epochs=10, batch_size=2, learning_rate=0.08)
        ai.TrainClassifier(clf, ds, cfg)
        after = clf.compute_loss("great day", "Happy")
        print(f"loss before: {before:.4f}")
        print(f"loss after:  {after:.4f}")
        print("predict:", clf.predict("great day"))


if __name__ == "__main__":
    main()
