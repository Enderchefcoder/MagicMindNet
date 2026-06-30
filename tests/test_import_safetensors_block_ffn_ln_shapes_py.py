import json
from pathlib import Path

import pytest

import magicmindnet as ai


def _export_chatbot(tmp_path: Path) -> Path:
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=1)
    path = tmp_path / "bot.mmn"
    ai.export(bot, "safetensors", str(path))
    return path


def test_import_safetensors_rejects_ln1_beta_shape_mismatch_py(tmp_path: Path):
    path = _export_chatbot(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tensors"]["blocks.0.ln1.beta"]["shape"] = [4, 4]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_model("safetensors", [str(path)])
    msg = str(exc.value).lower()
    assert "blocks.0.ln1.beta" in msg and "shape" in msg


def test_import_safetensors_rejects_ln2_beta_shape_mismatch_py(tmp_path: Path):
    path = _export_chatbot(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tensors"]["blocks.0.ln2.beta"]["shape"] = [4, 4]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_model("safetensors", [str(path)])
    msg = str(exc.value).lower()
    assert "blocks.0.ln2.beta" in msg and "shape" in msg


def test_import_safetensors_rejects_ffn_shape_mismatch_py(tmp_path: Path):
    path = _export_chatbot(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tensors"]["blocks.0.ffn"]["shape"] = [128, 8]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_model("safetensors", [str(path)])
    msg = str(exc.value).lower()
    assert "blocks.0.ffn" in msg and "shape" in msg


def test_import_safetensors_rejects_ln_gamma_shape_mismatch_py(tmp_path: Path):
    path = _export_chatbot(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tensors"]["blocks.0.ln1.gamma"]["shape"] = [4, 4]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_model("safetensors", [str(path)])
    msg = str(exc.value).lower()
    assert "blocks.0.ln1.gamma" in msg and "shape" in msg
