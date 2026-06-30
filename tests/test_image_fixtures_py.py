"""Image dataset fixture loaders."""

from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_image_gen_fixture_loads():
    ds = ai.DatasetImageGen(file=str(FIXTURES / "image_gen.json"))
    assert ds.rows == 1
    assert ds.format == "image_gen"
    assert ds.type_ == "image_gen"


def test_image_edit_fixture_loads():
    ds = ai.DatasetImageEdit(file=str(FIXTURES / "image_edit.json"))
    assert ds.rows == 1
    assert ds.format == "image_edit"
    assert ds.type_ == "image_edit"


def test_image_gen_repr_includes_rows():
    ds = ai.DatasetImageGen(file=str(FIXTURES / "image_gen.json"))
    text = repr(ds)
    assert "DatasetImageGen" in text
    assert "image_gen" in text
