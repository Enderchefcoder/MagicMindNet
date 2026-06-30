import json

import magicmindnet as ai


def test_dataset_classification_unique_labels(tmp_path):
    path = tmp_path / "cls.json"
    path.write_text(
        json.dumps(
            [
                {"text": "a", "tag": "Happy"},
                {"text": "b", "tag": "Sad"},
                {"text": "c", "tag": "Happy"},
            ]
        ),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(str(path), "text", "tag")
    assert ds.unique_labels() == ["Happy", "Sad"]
