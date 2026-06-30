import json
from pathlib import Path

import magicmindnet as ai


def test_dataset_classification_repr_shows_rows_and_labels(tmp_path: Path):
    path = tmp_path / "cls.json"
    path.write_text(
        json.dumps(
            [
                {"text": "a", "tag": "Happy"},
                {"text": "b", "tag": "Sad"},
            ]
        ),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(str(path), "text", "tag")
    text = repr(ds)
    assert "DatasetClassification" in text
    assert "rows=2" in text
    assert "Happy" in text
