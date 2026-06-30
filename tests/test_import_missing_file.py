from pathlib import Path

import pytest

import magicmindnet as ai


def test_import_model_safetensors_missing_file_raises():
    missing = Path("definitely_not_a_checkpoint_30.mmn")
    assert not missing.exists()
    with pytest.raises((OSError, RuntimeError, ValueError)):
        ai.import_model("safetensors", [str(missing)])
