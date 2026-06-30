import json
from pathlib import Path

import pytest

import magicmindnet as ai


def _assert_import_fails_missing_tensor(tmp_path: Path, tensor_key: str) -> None:
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=1)
    path = tmp_path / "bot.mmn"
    ai.export(bot, "safetensors", str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tensors"].pop(tensor_key, None)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_model("safetensors", [str(path)])
    assert tensor_key.lower() in str(exc.value).lower()


def test_import_safetensors_rejects_missing_block_ffn_py(tmp_path: Path):
    _assert_import_fails_missing_tensor(tmp_path, "blocks.0.ffn")


def test_import_safetensors_rejects_missing_block_attn_q_py(tmp_path: Path):
    _assert_import_fails_missing_tensor(tmp_path, "blocks.0.attn.q")


def test_import_safetensors_rejects_missing_block_attn_k_py(tmp_path: Path):
    _assert_import_fails_missing_tensor(tmp_path, "blocks.0.attn.k")


def test_import_safetensors_rejects_missing_block_ffn2_py(tmp_path: Path):
    _assert_import_fails_missing_tensor(tmp_path, "blocks.0.ffn2")


def test_import_safetensors_rejects_missing_block_ln1_gamma_py(tmp_path: Path):
    _assert_import_fails_missing_tensor(tmp_path, "blocks.0.ln1.gamma")
