import pytest

import magicmindnet as ai


def test_import_model_requires_at_least_one_file():
    with pytest.raises(ValueError, match="files required"):
        ai.import_model("safetensors", [])


def test_import_classifier_requires_at_least_one_file():
    with pytest.raises(ValueError, match="files required"):
        ai.import_classifier("safetensors", [])


def test_import_model_bin_requires_at_least_one_file():
    with pytest.raises(ValueError, match="files required"):
        ai.import_model("bin", [])
