import json
from pathlib import Path

import pytest

import magicmindnet as ai


def test_train_classifier_cuda_without_gpu_raises(tmp_path: Path):
    path = tmp_path / "cls.json"
    path.write_text(
        json.dumps([{"text": "hi", "tags": "A"}, {"text": "bye", "tags": "B"}]),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(file=str(path), text_col="text", tags_col="tags")
    clf = ai.Classifier.with_labels(["A", "B"], input_dim=16, seed=1)
    cfg = ai.TrainConfig(epochs=1, batch_size=1, cuda=True)
    with pytest.raises(ai.CUDAError):
        ai.TrainClassifier(clf, ds, cfg)
