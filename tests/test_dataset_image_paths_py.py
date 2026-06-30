"""DatasetImageGen / DatasetImageEdit path resolver APIs."""

from pathlib import Path

import pytest

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_image_gen_resolve_image_path_relative_to_manifest():
    ds = ai.DatasetImageGen(file=str(FIXTURES / "image_gen.json"))
    resolved = Path(ds.resolve_image_path("samples/cat.png"))
    assert resolved.name == "cat.png"
    assert resolved.parent.name == "samples"
    assert resolved.is_file()


def test_image_gen_image_path_at_and_prompt_at():
    ds = ai.DatasetImageGen(file=str(FIXTURES / "image_gen.json"))
    path = Path(ds.image_path_at(0))
    assert path.is_file()
    assert "cat" in path.name
    assert "cat" in ds.prompt_at(0).lower()


def test_image_edit_resolve_paths_and_row_accessors():
    ds = ai.DatasetImageEdit(file=str(FIXTURES / "image_edit.json"))
    image = Path(ds.resolve_image_path("samples/photo.png"))
    mask = Path(ds.resolve_mask_path("samples/mask.png"))
    assert image.is_file()
    assert mask.is_file()
    assert Path(ds.image_path_at(0)) == image
    assert Path(ds.mask_path_at(0)) == mask
    assert ds.prompt_at(0) == "remove the logo"


def test_image_path_at_out_of_range_raises():
    ds = ai.DatasetImageGen(file=str(FIXTURES / "image_gen.json"))
    with pytest.raises(IndexError):
        ds.image_path_at(1)
