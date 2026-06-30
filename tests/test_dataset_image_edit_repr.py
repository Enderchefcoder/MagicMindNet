import json
from pathlib import Path

import magicmindnet as ai


def test_dataset_image_edit_repr_shows_rows(tmp_path: Path):
    path = tmp_path / "image_edit.json"
    path.write_text(
        json.dumps(
            [
                {
                    "prompt": "remove background",
                    "image": "in.png",
                    "mask_image": "mask.png",
                },
            ]
        ),
        encoding="utf-8",
    )
    ds = ai.DatasetImageEdit(file=str(path))
    text = repr(ds)
    assert "DatasetImageEdit" in text
    assert "rows=1" in text
    assert 'format="image_edit"' in text or "format='image_edit'" in text
    assert "image_edit" in text
