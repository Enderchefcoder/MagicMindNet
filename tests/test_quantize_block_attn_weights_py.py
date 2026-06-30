import json
from pathlib import Path

import magicmindnet as ai


def _export_tensors(bot: ai.Chatbot, path: Path) -> dict:
    ai.export(bot, "safetensors", str(path))
    return json.loads(path.read_text(encoding="utf-8"))["tensors"]


def test_quantize_int8_changes_block_attn_out_weights_py(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=1)
    before = _export_tensors(bot, tmp_path / "before.mmn")
    ai.quantize(bot, "int8")
    after = _export_tensors(bot, tmp_path / "after.mmn")
    assert before["blocks.0.attn.out"]["data"] != after["blocks.0.attn.out"]["data"]


def test_quantize_int4_changes_block_attn_v_weights_py(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=1)
    before = _export_tensors(bot, tmp_path / "before.mmn")
    ai.quantize(bot, "int4")
    after = _export_tensors(bot, tmp_path / "after.mmn")
    assert before["blocks.0.attn.v"]["data"] != after["blocks.0.attn.v"]["data"]


def test_quantize_int8_changes_block_attn_k_weights_py(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=1)
    before = _export_tensors(bot, tmp_path / "before.mmn")
    ai.quantize(bot, "int8")
    after = _export_tensors(bot, tmp_path / "after.mmn")
    assert before["blocks.0.attn.k"]["data"] != after["blocks.0.attn.k"]["data"]


def test_quantize_int4_changes_block_attn_out_weights_py(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=1)
    before = _export_tensors(bot, tmp_path / "before.mmn")
    ai.quantize(bot, "int4")
    after = _export_tensors(bot, tmp_path / "after.mmn")
    assert before["blocks.0.attn.out"]["data"] != after["blocks.0.attn.out"]["data"]


def test_quantize_int4_changes_block_ffn2_weights_py(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=1)
    before = _export_tensors(bot, tmp_path / "before.mmn")
    ai.quantize(bot, "int4")
    after = _export_tensors(bot, tmp_path / "after.mmn")
    assert before["blocks.0.ffn2"]["data"] != after["blocks.0.ffn2"]["data"]


def test_quantize_int8_changes_block_attn_q_weights_py(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=1)
    before = _export_tensors(bot, tmp_path / "before.mmn")
    ai.quantize(bot, "int8")
    after = _export_tensors(bot, tmp_path / "after.mmn")
    assert before["blocks.0.attn.q"]["data"] != after["blocks.0.attn.q"]["data"]


def test_quantize_int4_changes_block_attn_k_weights_py(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=1)
    before = _export_tensors(bot, tmp_path / "before.mmn")
    ai.quantize(bot, "int4")
    after = _export_tensors(bot, tmp_path / "after.mmn")
    assert before["blocks.0.attn.k"]["data"] != after["blocks.0.attn.k"]["data"]


def test_quantize_int4_changes_block_attn_q_weights_py(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=1)
    before = _export_tensors(bot, tmp_path / "before.mmn")
    ai.quantize(bot, "int4")
    after = _export_tensors(bot, tmp_path / "after.mmn")
    assert before["blocks.0.attn.q"]["data"] != after["blocks.0.attn.q"]["data"]


def test_quantize_int8_changes_block_ffn2_weights_py(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=1)
    before = _export_tensors(bot, tmp_path / "before.mmn")
    ai.quantize(bot, "int8")
    after = _export_tensors(bot, tmp_path / "after.mmn")
    assert before["blocks.0.ffn2"]["data"] != after["blocks.0.ffn2"]["data"]
