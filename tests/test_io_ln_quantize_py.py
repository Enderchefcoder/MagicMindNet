"""LayerNorm quantize parity: non-default γ/β mutate under int8/int4."""

import json
from pathlib import Path

import pytest

import magicmindnet as ai
from conftest import (
    load_checkpoint,
    tamper_tensor_entry_first_f32,
    tensor_entry_first_f32,
)


def test_int8_quantize_changes_ln1_gamma_when_non_default(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=1)
    path = tmp_path / "bot.mmn"
    ai.export(bot, "safetensors", str(path))
    payload = load_checkpoint(path)
    tamper_tensor_entry_first_f32(payload["tensors"]["blocks.0.ln1.gamma"], 1.37)
    path.write_text(json.dumps(payload), encoding="utf-8")
    bot2 = ai.import_model("safetensors", [str(path)])
    before = tensor_entry_first_f32(payload["tensors"]["blocks.0.ln1.gamma"])
    ai.quantize(bot2, "int8")
    out = tmp_path / "quant.mmn"
    ai.export(bot2, "safetensors", str(out))
    after = tensor_entry_first_f32(
        load_checkpoint(out)["tensors"]["blocks.0.ln1.gamma"]
    )
    expected = round(before * 127.0) / 127.0
    assert after == pytest.approx(expected, rel=1e-5, abs=1e-5)
    assert abs(after - 1.37) > 1e-6


def test_int4_quantize_changes_ln2_beta_when_non_default(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=1)
    path = tmp_path / "bot.mmn"
    ai.export(bot, "safetensors", str(path))
    payload = load_checkpoint(path)
    tamper_tensor_entry_first_f32(payload["tensors"]["blocks.0.ln2.beta"], 0.25)
    path.write_text(json.dumps(payload), encoding="utf-8")
    bot2 = ai.import_model("safetensors", [str(path)])
    before = tensor_entry_first_f32(payload["tensors"]["blocks.0.ln2.beta"])
    ai.quantize(bot2, "int4")
    out = tmp_path / "quant.mmn"
    ai.export(bot2, "safetensors", str(out))
    after = tensor_entry_first_f32(
        load_checkpoint(out)["tensors"]["blocks.0.ln2.beta"]
    )
    expected = round(before * 15.0) / 15.0
    assert after == pytest.approx(expected, rel=1e-5, abs=1e-5)
    assert abs(after - 0.25) > 1e-6
