import json
from pathlib import Path

import magicmindnet as ai


def test_dataset_image_gen_repr_shows_rows(tmp_path: Path):
    path = tmp_path / "image_gen.json"
    path.write_text(
        json.dumps(
            [
                {"prompt": "a cat", "image": "cat.png"},
                {"prompt": "a dog", "image": "dog.png", "negative_prompt": "blur"},
            ]
        ),
        encoding="utf-8",
    )
    ds = ai.DatasetImageGen(file=str(path))
    text = repr(ds)
    assert "DatasetImageGen" in text
    assert "rows=2" in text
    assert 'format="image_gen"' in text or "format='image_gen'" in text
    assert "image_gen" in text
