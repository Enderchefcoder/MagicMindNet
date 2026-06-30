import json
from pathlib import Path

import magicmindnet as ai


def test_dataset_image_gen_exposes_format_getter(tmp_path: Path):
    path = tmp_path / "gen.json"
    path.write_text(json.dumps([{"prompt": "x", "image": "a.png"}]), encoding="utf-8")
    ds = ai.DatasetImageGen(file=str(path))
    assert ds.format == "image_gen"
    assert ds.type_ == "image_gen"


def test_dataset_image_edit_exposes_format_getter(tmp_path: Path):
    path = tmp_path / "edit.json"
    path.write_text(
        json.dumps([{"prompt": "x", "image": "a.png", "mask_image": "m.png"}]),
        encoding="utf-8",
    )
    ds = ai.DatasetImageEdit(file=str(path))
    assert ds.format == "image_edit"
    assert ds.type_ == "image_edit"
