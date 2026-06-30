import json
from pathlib import Path

import magicmindnet as ai


def test_unique_labels_sorted_deduped(tmp_path: Path):
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            [
                {"text": "a", "tag": "B"},
                {"text": "b", "tag": "A"},
                {"text": "c", "tag": "B"},
            ]
        ),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(str(path), "text", "tag")
    assert ds.unique_labels() == ["A", "B"]
