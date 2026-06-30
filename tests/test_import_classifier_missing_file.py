from pathlib import Path

import pytest

import magicmindnet as ai


def test_import_classifier_missing_file_raises():
    missing = Path("definitely_not_a_classifier_39.mmn")
    assert not missing.exists()
    with pytest.raises((OSError, RuntimeError, ValueError)):
        ai.import_classifier("safetensors", [str(missing)])
