import json
import math
from pathlib import Path

import magicmindnet as ai


def test_classifier_compute_mean_loss_is_finite_before_train(tmp_path: Path):
    path = tmp_path / "cls.json"
    path.write_text(
        json.dumps(
            [
                {"text": "good", "tag": "pos"},
                {"text": "bad", "tag": "neg"},
            ]
        ),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(str(path), "text", "tag")
    clf = ai.Classifier.from_classification(ds, input_dim=16, seed=1)
    loss = clf.compute_mean_loss(ds)
    assert math.isfinite(loss)
    assert loss > 0.0
